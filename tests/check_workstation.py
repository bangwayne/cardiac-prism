#!/usr/bin/env python3
"""Exercise a running, isolated workstation with the released model and sample.

This integration check creates studies and changes the server's inference
device. It refuses a nonempty workspace unless explicitly permitted. Never run
it against a workstation being used for annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile

import numpy as np
import SimpleITK as sitk


def digest(value):
    return hashlib.sha256(value).hexdigest()


class API:
    def __init__(self, base, output, timeout):
        self.base = base.rstrip('/')
        self.output = output
        self.timeout = timeout
        self.jobs = []
        self.created = []

    def request(self, path, payload=None, binary=False, method=None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {'Content-Type': 'application/json'} if data is not None else {}
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                return body if binary else json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            raise RuntimeError(f'{request.get_method()} {path}: HTTP {exc.code}: {detail}') from exc

    def upload(self, path, source, name=None):
        boundary = 'cardiacprism-' + uuid.uuid4().hex
        filename = name or source.name
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        body += source.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        request = urllib.request.Request(self.base + path, data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Upload {path}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
        self.created.append(result['id'])
        return result['id']

    def view(self, study, t=0, z=0):
        return self.request(f'/api/projects/{study}?time={t}&slice={z}')

    def post(self, study, action, **fields):
        return self.request(f'/api/projects/{study}/{action}',
                            {'revision': self.view(study)['revision'], **fields})

    def run(self, study, action, **fields):
        started = time.monotonic()
        job = self.post(study, action, **fields)
        job_id = job['id']
        seen_phases = set()
        deadline = started + self.timeout
        next_report = started
        while time.monotonic() < deadline:
            job = self.request('/api/jobs/' + job_id)
            seen_phases.add(job.get('phase', 'unknown'))
            if job['state'] not in ('queued', 'running'):
                break
            if time.monotonic() >= next_report:
                print(json.dumps({'action': action, 'state': job['state'], 'phase': job.get('phase'),
                                  'completed_slices': job.get('completed_slices'),
                                  'total_slices': job.get('total_slices')}), flush=True)
                next_report = time.monotonic() + 15
            time.sleep(.2)
        else:
            self.request('/api/jobs/' + job_id + '/cancel', {}, method='POST')
            raise TimeoutError(f'{action} exceeded {self.timeout} seconds')
        elapsed = time.monotonic() - started
        self.jobs.append({'action': action, 'seconds': elapsed, 'observed_phases': sorted(seen_phases), **job})
        if job['state'] != 'completed':
            raise AssertionError(f'{action} did not complete: {job}')
        print(json.dumps({'completed': action, 'seconds': round(elapsed, 3),
                          'gpu_completed': job.get('gpu_completed')}), flush=True)
        return job

    def masks(self, study):
        shape = self.view(study)['shape']
        frames = []
        for t in range(shape[0]):
            content = self.request(f'/api/projects/{study}/masks?time={t}', binary=True)
            frames.append(np.frombuffer(content, dtype=np.uint8).reshape((3, *shape[1:])).copy())
        return np.stack(frames, axis=1)

    def backup(self, study, name):
        destination = self.output / f'{name}.zip'
        with urllib.request.urlopen(self.base + f'/api/projects/{study}/export?kind=study',
                                    timeout=self.timeout) as response, destination.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        return destination


def backup_signature(path):
    """Hash actual prior arrays, independent of ZIP filenames or compression."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('project.json'))
        inventory = json.loads(archive.read('backup.json'))
        for name, expected in inventory['files'].items():
            sha = hashlib.sha256()
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    sha.update(chunk)
            assert sha.hexdigest() == expected, f'Backup integrity mismatch: {name}'
        grouped = {}
        contexts = {}
        priors = {}
        for record in manifest['contexts']:
            key = ':'.join(map(str, record['key']))
            contexts[key] = {field: record.get(field) for field in ('points', 'iteration', 'mode', 'mask_prior')}
            if 'prior_key' in record:
                filename = record.get('prior_file', manifest['state_file'])
                grouped.setdefault(filename, []).append((key, record['prior_key']))
        for filename, records in grouped.items():
            # A seek inside nested ZIP streams can repeatedly scan a large outer
            # member. Spool each NPZ once and keep array validation bounded.
            with tempfile.TemporaryFile() as cache:
                with archive.open(filename) as stream:
                    shutil.copyfileobj(stream, cache, length=1024 * 1024)
                cache.seek(0)
                with np.load(cache, allow_pickle=False) as arrays:
                    for key, array_name in records:
                        values = arrays[array_name]
                        assert np.isfinite(values).all(), f'Nonfinite prior at {key}'
                        priors[key] = {'shape': list(values.shape), 'dtype': str(values.dtype),
                                       'sha256': digest(values.tobytes())}
        return {'priors': priors, 'contexts': contexts, 'locked_slices': manifest.get('locked_slices', []),
                'source_geometry': manifest['source_geometry'],
                'notes': json.loads(archive.read('study-notes.json')),
                'source_sha256': inventory['files'][manifest['source_file']],
                'events': manifest['events'], 'manifest': manifest}


def assert_geometry(first, second):
    assert first.GetSize() == second.GetSize()
    assert first.GetDimension() == second.GetDimension()
    for field in ('GetSpacing', 'GetOrigin', 'GetDirection'):
        np.testing.assert_allclose(getattr(first, field)(), getattr(second, field)(), rtol=0, atol=1e-5)


def check_equal(first, second, message):
    np.testing.assert_array_equal(first, second, err_msg=message)


def smoke(api, source, output, gpu_ids, cpu_check, checks):
    image = sitk.ReadImage(str(source))
    assert image.GetDimension() == 4 and image.GetSize()[2] >= 2 and image.GetSize()[3] >= 2
    start_z = max(0, image.GetSize()[2] // 2 - 1)
    crop = image[:, :, start_z:start_z + 2, 0:2]
    crop_path = output / 'integration-cine.nii.gz'
    sitk.WriteImage(crop, str(crop_path))
    crop = sitk.ReadImage(str(crop_path))
    study = api.upload('/api/projects', crop_path)
    view = api.view(study)
    assert view['shape'][:2] == [2, 2]
    checks.append('4D upload and geometry')

    target_device = 'cuda:0' if gpu_ids else 'cpu'
    api.request('/api/model/device', {'device': target_device})
    api.run(study, 'inference', time=0, slice=0, organ=2, mode='organ')
    initial = api.masks(study)
    assert initial[1, 0, 0].any(), 'Released model produced an empty MYO initialization on the sample crop'
    before_click = initial.copy()
    y, x = np.argwhere(initial[1, 0, 0])[len(np.argwhere(initial[1, 0, 0])) // 2]
    api.run(study, 'inference', time=0, slice=0, organ=2, mode='organ_click',
            point={'x': int(x), 'y': int(y), 'positive': True})
    positive = api.masks(study)
    assert api.view(study)['structures'][1]['points'][-1]['positive'] is True
    y2, x2 = np.argwhere(positive[1, 0, 0])[0]
    api.run(study, 'inference', time=0, slice=0, organ=2, mode='organ_click',
            point={'x': int(x2), 'y': int(y2), 'positive': False})
    negative = api.masks(study)
    assert api.view(study)['structures'][1]['points'][-1]['positive'] is False
    check_equal(negative[:, 1], before_click[:, 1], 'Clicks changed another time frame')
    check_equal(negative[:, :, 1], before_click[:, :, 1], 'Clicks changed another layer')
    check_equal(negative[[0, 2]], before_click[[0, 2]], 'Clicks changed another structure')
    assert not np.array_equal(positive[1, 0, 0], negative[1, 0, 0]), 'Negative click did not change the sample mask'
    checks.append('Real positive/negative click inference and 2D scope')

    if cpu_check and gpu_ids:
        api.request('/api/model/device', {'device': 'cpu'})
        api.run(study, 'inference', time=0, slice=0, organ=2, mode='organ')
        cpu = api.masks(study)
        assert cpu[1, 0, 0].any()
        disagreement = int(np.count_nonzero(cpu[1, 0, 0] != initial[1, 0, 0]))
        checks.append({'CPU/GPU device switching': True, 'initialization_mask_disagreement_pixels': disagreement})
        api.request('/api/model/device', {'device': target_device})

    before_brush = api.masks(study)
    points = [[1, 1]]
    api.post(study, 'brush', time=0, slice=0, organ=1, points=points, radius=0, erase=False)
    api.post(study, 'brush', time=0, slice=0, organ=2, points=points, radius=0, erase=False)
    painted = api.masks(study)
    assert list(painted[:, 0, 0, 1, 1]) == [0, 1, 0]
    expected = before_brush.copy()
    expected[:, 0, 0, 1, 1] = [0, 1, 0]
    check_equal(painted, expected, 'A one-pixel brush changed other pixels')
    api.post(study, 'brush', time=0, slice=0, organ=2, points=points, radius=0, erase=True)
    erased = api.masks(study)
    expected[:, 0, 0, 1, 1] = 0
    check_equal(erased, expected, 'A one-pixel eraser changed other pixels')
    api.post(study, 'undo')
    check_equal(api.masks(study), painted, 'Undo did not restore the brush result')
    api.post(study, 'redo')
    check_equal(api.masks(study), erased, 'Redo did not restore the eraser result')
    checks.append('One-pixel brush priority, erase, undo and redo')

    api.post(study, 'slice-locks', times=[0], slices=[0], locked=True)
    protected = api.masks(study)[:, 0, 0].copy()
    locked_backup = backup_signature(api.backup(study, 'smoke-before-locked-batch'))
    fields = {'gpu_ids': gpu_ids} if gpu_ids else {}
    job = api.run(study, 'segment-all-time-frames', **fields)
    assert job['completed_slices'] == 3
    assert job['skipped_locked_slices'] == [[0, 0]]
    check_equal(api.masks(study)[:, 0, 0], protected, 'Batch segmentation changed the locked slice')
    after_locked = backup_signature(api.backup(study, 'smoke-after-locked-batch'))
    for key, value in locked_backup['contexts'].items():
        if key.startswith('0:0:'):
            assert after_locked['contexts'][key] == value
            assert after_locked['priors'].get(key) == locked_backup['priors'].get(key)
    checks.append('Batch inference skips local locks and preserves prompts/priors')

    before_propagation = api.masks(study)
    request = {'slice': 1, 'reference_times': [0], 'organs': [1, 2, 3],
               'target_times': [1], 'overwrite_edited': True, **fields}
    api.post(study, 'slice-locks', times=[1], slices=[1], locked=True)
    preview = api.post(study, 'propagation-preview', **request)
    assert preview['total_predictions'] == 0
    assert preview['skipped_locked_slices'] == [[1, 1]]
    api.post(study, 'slice-locks', times=[1], slices=[1], locked=False)
    api.run(study, 'propagate-time', **request)
    propagated = api.masks(study)
    check_equal(propagated[:, :, 0], before_propagation[:, :, 0], 'Propagation changed another layer')
    check_equal(propagated[:, 0, 1], before_propagation[:, 0, 1], 'Propagation changed its reference')
    check_equal(propagated[:, 0, 0], protected, 'Propagation changed a locked slice')
    api.post(study, 'undo')
    check_equal(api.masks(study), before_propagation, 'Propagation undo did not restore the full prior result')
    api.post(study, 'redo')
    check_equal(api.masks(study), propagated, 'Propagation redo did not restore the result')
    checks.append('Temporal propagation, protected target preview, reference scope and undo/redo')

    api.post(study, 'save')
    notes = api.request(f'/api/projects/{study}/notes')
    note = api.request(f'/api/projects/{study}/notes', {'revision': notes['revision'],
        'text': 'Integration validation: corrected MYO and protected frame 1, layer 1.', 'review_status': 'in_progress'})
    backup_path = api.backup(study, 'smoke-complete')
    original_signature = backup_signature(backup_path)
    restored = api.upload('/api/projects/import-backup', backup_path)
    check_equal(api.masks(restored), propagated, 'Full backup import changed masks')
    assert api.view(restored)['locked_slices'] == [[0, 0]]
    assert api.view(restored)['undo_count'] == 0
    assert api.request(f'/api/projects/{restored}/notes') == note
    restored_signature = backup_signature(api.backup(restored, 'smoke-restored'))
    for key in ('priors', 'contexts', 'locked_slices', 'source_geometry', 'notes', 'source_sha256', 'events'):
        assert original_signature[key] == restored_signature[key], f'Backup round trip changed {key}'
    label_path = output / 'smoke-labelmap.nii.gz'
    label_path.write_bytes(api.request(f'/api/projects/{restored}/export?kind=labelmap', binary=True))
    exported = sitk.ReadImage(str(label_path))
    assert_geometry(crop, exported)
    check_equal(sitk.GetArrayFromImage(exported), sum((label + 1) * propagated[label] for label in range(3)),
                'Combined labelmap differs from independent masks')
    checks.append('Save, verified full backup restore, notes/locks/priors/activity and 4D labelmap geometry')
    return {'study': study, 'restored_study': restored, 'shape': view['shape'],
            'positive_mask_changes': int(np.count_nonzero(positive != before_click)),
            'negative_mask_changes': int(np.count_nonzero(positive != negative)),
            'mask_sha256': digest(propagated.tobytes()), 'prior_count': len(original_signature['priors'])}


def full_cine(api, source, gpu_ids, checks):
    study = api.upload('/api/projects', source)
    shape = api.view(study)['shape']
    single = api.run(study, 'segment-all-time-frames', gpu_ids=[gpu_ids[0]])
    first_masks = api.masks(study)
    first = backup_signature(api.backup(study, 'full-single-gpu'))
    parallel = api.run(study, 'segment-all-time-frames', gpu_ids=gpu_ids)
    second_masks = api.masks(study)
    second = backup_signature(api.backup(study, 'full-multiple-gpu'))
    expected_count = shape[0] * shape[1]
    assert single['completed_slices'] == parallel['completed_slices'] == expected_count
    assert len(first['priors']) == len(second['priors']) == expected_count * 3
    check_equal(first_masks, second_masks, 'Single-GPU and multiple-GPU masks differ')
    assert first['priors'] == second['priors'], 'Single-GPU and multiple-GPU priors differ'
    assert first['contexts'] == second['contexts'], 'Single-GPU and multiple-GPU contexts differ'
    assert sum(parallel['gpu_completed'].values()) == expected_count
    assert all(parallel['gpu_completed'].get(str(gpu), 0) > 0 for gpu in gpu_ids)
    assert not np.any(second_masks.sum(axis=0) > 1), 'Joint inference left overlapping masks'
    checks.append('Full-cine real single/multiple-GPU masks and all probability priors are bit-identical')
    return {'study': study, 'shape': shape, 'planes': expected_count,
            'prior_count': len(first['priors']), 'mask_sha256': digest(first_masks.tobytes()),
            'prior_manifest_sha256': digest(json.dumps(first['priors'], sort_keys=True).encode()),
            'single_gpu_job': single, 'multiple_gpu_job': parallel}


def verify_restart(api, previous_path):
    """Check a completed run after its isolated server has been restarted."""
    previous = json.loads(previous_path.read_text())
    assert previous['status'] == 'passed', 'The original validation must pass before a restart check'
    old = backup_signature(previous_path.parent / 'smoke-complete.zip')
    for study in (previous['smoke']['study'], previous['smoke']['restored_study']):
        view = api.view(study)
        assert view['undo_count'] == view['redo_count'] == 0, 'Restart must clear session history'
        assert not view['dirty'], 'Restarted study should reflect its saved revision'
        assert digest(api.masks(study).tobytes()) == previous['smoke']['mask_sha256']
        signature = backup_signature(api.backup(study, 'restart-' + study))
        for key in ('priors', 'contexts', 'locked_slices', 'source_geometry', 'notes', 'source_sha256', 'events'):
            assert signature[key] == old[key], f'Restart changed {key}'
    if 'full_cine' in previous:
        study = previous['full_cine']['study']
        assert digest(api.masks(study).tobytes()) == previous['full_cine']['mask_sha256']
        assert api.view(study)['undo_count'] == 0
        signature = backup_signature(api.backup(study, 'restart-full-cine'))
        assert digest(json.dumps(signature['priors'], sort_keys=True).encode()) == previous['full_cine']['prior_manifest_sha256']
    result = {'status': 'passed', 'checks': ['Saved masks, geometry, notes, locks, prompts, activity and priors survive restart',
                                          'Session undo/redo is cleared on restart'],
              'studies': previous['created_studies']}
    (api.output / 'restart-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Isolated test service, for example http://127.0.0.1:8517')
    parser.add_argument('--image', type=Path, default=Path('demo-data/images/patient001_4d.nii.gz'))
    parser.add_argument('--output', type=Path, required=True, help='Directory outside the published source tree')
    parser.add_argument('--gpu-ids', help='Comma-separated GPU IDs offered by /api/gpus; omit for CPU-only checks')
    parser.add_argument('--cpu-check', action='store_true', help='Also run CPU inference and restore the GPU device')
    parser.add_argument('--full-cine', action='store_true', help='Compare a full cine on one and multiple GPUs')
    parser.add_argument('--resume-report', type=Path, help='After restarting the test server, verify the studies from this completed report')
    parser.add_argument('--allow-existing-studies', action='store_true', help='Allow a disposable test workspace that already has studies')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(value) for value in args.gpu_ids.split(',')] if args.gpu_ids else []
    if args.full_cine and len(gpu_ids) < 2:
        parser.error('--full-cine requires at least two --gpu-ids')
    api = API(args.url, args.output, args.timeout)
    if args.resume_report:
        verify_restart(api, args.resume_report)
        return
    report = {'status': 'running', 'checks': [], 'started_at': time.time(), 'gpu_ids': gpu_ids}
    try:
        report['health_before'] = api.request('/api/health')
        existing = api.request('/api/projects')
        if existing and not args.allow_existing_studies:
            raise RuntimeError('Refusing a nonempty workspace. Start an isolated test server or explicitly allow disposable existing studies.')
        assert args.image.is_file(), f'Sample image not found: {args.image}'
        report['input_sha256'] = digest(args.image.read_bytes())
        report['smoke'] = smoke(api, args.image, args.output, gpu_ids, args.cpu_check, report['checks'])
        if args.full_cine:
            report['full_cine'] = full_cine(api, args.image, gpu_ids, report['checks'])
        report['health_after'] = api.request('/api/health')
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['created_studies'] = api.created
        report['jobs'] = api.jobs
        report['elapsed_seconds'] = time.time() - report['started_at']
        (args.output / 'workstation-validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                          'output': str(args.output / 'workstation-validation.json')}), flush=True)


if __name__ == '__main__':
    main()
