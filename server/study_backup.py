"""Portable, verified study snapshots including image, editable state and notes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

import numpy as np
import SimpleITK as sitk

from .domain import geometry, locked_slices_from_manifest
from .study_notes import read_notes

FORMAT = 'kprism-study'
MAX_BYTES = 8 * 1024 ** 3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def referenced_files(manifest):
    names = {manifest['source_file'], manifest['state_file']}
    names.update(item['file'] for item in manifest.get('mask_layers', {}).values())
    names.update(item['prior_file'] for item in manifest['contexts'] if 'prior_file' in item)
    if manifest.get('reference_file'):
        names.add(manifest['reference_file'])
    for name in names:
        if Path(name).name != name or '\\' in name or name in ('', '.', '..'):
            raise ValueError('Invalid study backup path.')
    return names


def create_backup(project):
    project.save()
    descriptor, filename = tempfile.mkstemp(prefix='study-backup-', suffix='.zip', dir=project.directory.parent.parent)
    os.close(descriptor)
    path = Path(filename)
    try:
        # A save can prune obsolete chunks. Hold only its save lock while copying
        # the immutable snapshot; editing/inference does not take this lock.
        with project.save_lock:
            manifest = json.loads((project.directory / 'project.json').read_text())
            payload = {'project.json': json.dumps(manifest, ensure_ascii=False).encode(),
                       'study-notes.json': json.dumps(read_notes(project.directory), ensure_ascii=False).encode()}
            hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in sorted(referenced_files(manifest)):
                    file = project.directory / name
                    hashes[name] = sha(file)
                    archive.write(file, name)
                for name, value in payload.items():
                    archive.writestr(name, value)
                archive.writestr('backup.json', json.dumps(dict(format=FORMAT, version=1, files=hashes,
                    annotation_revision=manifest['revision'], undo_history_included=False)))
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def import_backup(store, path):
    staging = Path(tempfile.mkdtemp(prefix='.import-', dir=store.directory))
    destination = None
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or len(names) > 4096 or sum(e.file_size for e in entries) > MAX_BYTES:
                raise ValueError('The study backup has duplicate entries or exceeds the size limit.')
            if any(Path(name).name != name or '\\' in name or name in ('', '.', '..') for name in names):
                raise ValueError('Invalid study backup path.')
            if 'backup.json' not in names:
                raise ValueError('Choose a full study backup ZIP. Extract a mask-only ZIP and import its NIfTI masks separately.')
            description = json.loads(archive.read('backup.json'))
            if not isinstance(description, dict) or description.get('format') != FORMAT or description.get('version') != 1:
                raise ValueError('Choose a Cardiac-Prism full study backup (.zip).')
            expected = description['files']
            if not isinstance(expected, dict) or not all(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64 for name, digest in expected.items()):
                raise ValueError('Invalid study backup file inventory.')
            if set(names) != set(expected) | {'backup.json'}:
                raise ValueError('The study backup file inventory is inconsistent.')
            for name in expected:
                with archive.open(name) as source, (staging / name).open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if sha(staging / name) != expected[name]:
                    raise ValueError('Study backup integrity verification failed: ' + name)
        manifest = json.loads((staging / 'project.json').read_text())
        if set(expected) != referenced_files(manifest) | {'project.json', 'study-notes.json'}:
            raise ValueError('The study backup contains missing or unreferenced files.')
        note = read_notes(staging)
        if not isinstance(note.get('text'), str) or len(note['text']) > 5000 or note.get('review_status') not in ('in_progress', 'complete'):
            raise ValueError('Invalid study note in backup.')
        image = sitk.ReadImage(str(staging / manifest['source_file']))
        if geometry(image) != manifest['source_geometry'] or np.prod(image.GetSize(), dtype=np.int64) > 128_000_000:
            raise ValueError('Study backup image geometry differs or exceeds the voxel limit.')
        # Protection is part of the portable state. Reject invalid or ambiguous
        # frame/layer indices before decompressing the potentially large priors.
        frames = image.GetSize()[3] if image.GetDimension() == 4 else 1
        locked_slices_from_manifest(manifest, frames, image.GetSize()[2])
        # Bound nested NPZ expansion and validate each referenced prior without
        # retaining all 1,440 priors in RAM. Regular reopening remains lazy.
        expanded = 0
        for name in referenced_files(manifest) - {manifest['source_file']}:
            with zipfile.ZipFile(staging / name) as arrays:
                expanded += sum(item.file_size for item in arrays.infolist())
                if expanded > MAX_BYTES:
                    raise ValueError('Study backup arrays exceed the size limit.')
        for record in manifest['contexts']:
            if 'prior_key' in record:
                name = record.get('prior_file', manifest['state_file'])
                with np.load(staging / name, allow_pickle=False) as arrays:
                    value = arrays[record['prior_key']]
                    if value.shape != tuple(record.get('prior_shape', (1, 512, 512))) or value.dtype.kind != 'f' or not np.isfinite(value).all():
                        raise ValueError('Invalid probability prior in study backup.')
        project_id = uuid.uuid4().hex
        manifest['id'] = project_id
        (staging / 'project.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        destination = store.directory / project_id
        staging.rename(destination)
        project = store.get(project_id)
        return project
    except (zipfile.BadZipFile, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        raise ValueError('The study backup is invalid or incomplete.') from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if destination is not None and destination.name not in store.projects:
            shutil.rmtree(destination, ignore_errors=True)
