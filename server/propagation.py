"""Same-layer temporal propagation as one sparse, revision-safe transaction."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import logging
import time
import uuid

import numpy as np

from .annotation_progress import summary
from .domain import ORGANS, now
from .study_notes import read_notes

log = logging.getLogger(__name__)


class PropagationMixin:
    def propagation_preview(self, project, request):
        """Compute eligibility from current annotations/history without reading priors."""
        with project.lock:
            project.check_revision(request['revision'])
            if project.pending_job:
                raise ValueError('Wait for the active inference or cancel it before propagating.')
            if read_notes(project.directory)['review_status'] == 'complete':
                raise ValueError('Reopen this completed study before propagating annotations.')
            frames, depth = project.voxels.shape[:2]
            if frames < 2:
                raise ValueError('Temporal propagation needs a 4D study with at least two time frames.')
            z = request['slice']
            project.check_slice(0, z)
            references = request['reference_times']
            labels = request['organs']
            if not 1 <= len(references) <= 3 or len(set(references)) != len(references):
                raise ValueError('Select one to three distinct reference time frames.')
            if not labels or len(set(labels)) != len(labels):
                raise ValueError('Select distinct structures to propagate.')
            for t in references:
                project.check_slice(t, z)
            for label in labels:
                project.check_slice(0, z, label)
            labels = sorted(labels)
            times = request.get('target_times')
            if times is None:
                times = list(range(frames))
            if not times or len(times) != len(set(times)):
                raise ValueError('Choose distinct target time frames.')
            for t in times:
                project.check_slice(t, z)
            progress = summary(project)
            reference_organs = [{'time': t, 'organs': [label for label in ORGANS
                                if np.any(project.masks[label][t, z])]} for t in references]
            support = {label: [item['time'] for item in reference_organs if label in item['organs']]
                       for label in labels}
            warnings = []
            for label, sources in support.items():
                if len(sources) < len(references):
                    warnings.append(f'{ORGANS[label]["short"]} has a nonempty mask in {len(sources)} of '
                                    f'{len(references)} selected references; only those references can guide this structure.')
            if not progress['history_complete']:
                warnings.append('Some edit history is unavailable. Existing nonempty target annotations are protected by default.')
            eligible, skipped = [], []
            for t in sorted(times):
                selected = []
                edited = progress['edited_labels'][t * depth + z]
                for label in labels:
                    reason = ('reference' if t in references else
                              'slice_locked' if (t, z) in project.locked_slices else
                              'locked' if project.locks[label] else
                              'no_reference_mask' if not support[label] else
                              'edited' if not request.get('overwrite_edited', False) and
                              (edited & (1 << (label - 1)) or not progress['history_complete'] and
                               np.any(project.masks[label][t, z])) else None)
                    if reason:
                        skipped.append({'time': t, 'organ': label, 'reason': reason})
                    else:
                        selected.append(label)
                if selected:
                    eligible.append({'time': t, 'organs': selected})
            return {'revision': project.revision, 'slice': z, 'reference_times': list(references),
                    'organs': labels, 'eligible': eligible, 'skipped': skipped,
                    'total_frames': len(eligible), 'total_predictions': sum(len(item['organs']) for item in eligible),
                    'reference_organs': reference_organs, 'warnings': warnings,
                    'skipped_locked_frames': sorted(set(times) & project.locked_frames - set(references)),
                    'skipped_locked_slices': [[t, z] for t in sorted(times)
                                             if (t, z) in project.locked_slices and t not in references],
                    'overwrite_edited': request.get('overwrite_edited', False)}

    def submit_propagation(self, project, request):
        gpu_ids = request.get('gpu_ids')
        if gpu_ids is not None:
            self.gpu_pool.validate(gpu_ids)
        with project.lock:
            preview = self.propagation_preview(project, request)
            if not preview['total_predictions']:
                raise ValueError('No eligible target annotations. Choose a nonempty reference mask, unlock structures, or review protected edits.')
            z = request['slice']
            keys = [(entry['time'], z, label) for entry in preview['eligible'] for label in entry['organs']]
            snapshot = project.snapshot_planes(keys, 'Propagate reference masks through time on this layer')
            # Freeze supports and protected pixels before dispatch. Targets are
            # independent; inferred masks never become another target's support.
            reference_masks = {label: [(t, project.masks[label][t, z].copy()) for t in request['reference_times']
                               if np.any(project.masks[label][t, z])] for label in preview['organs']}
            protected = {}
            for entry in preview['eligible']:
                mask = np.zeros(project.voxels.shape[2:], bool)
                for label in ORGANS:
                    if label not in entry['organs']:
                        mask |= project.masks[label][entry['time'], z].astype(bool)
                protected[entry['time']] = mask
            job = {'id': uuid.uuid4().hex, 'project_id': project.id, 'time': preview['eligible'][0]['time'],
                   'slice': z, 'organ': 0, 'mode': 'propagation', 'organs': preview['organs'],
                   'reference_times': list(request['reference_times']), 'state': 'queued', 'created_at': now(),
                   'seconds': 0, 'base_revision': project.revision, 'error': None,
                   'completed_frames': 0, 'total_frames': preview['total_frames'],
                   'completed_slices': 0, 'total_slices': preview['total_frames'], 'current_slice': z,
                   'current_time': preview['eligible'][0]['time'], 'gpu_ids': gpu_ids,
                   'execution_devices': [], 'gpu_completed': {}, 'preview': preview}
            job.update(phase='queued', phase_started_at=job['created_at'], ready_gpu_ids=[],
                       preparing_gpu_ids=[], result_visibility='on_completion')
            job['target_times'] = [item['time'] for item in preview['eligible']]
            job['target_slices'] = [[t, z] for t in job['target_times']]
            job['skipped_locked_frames'] = preview['skipped_locked_frames']
            job['skipped_locked_slices'] = preview['skipped_locked_slices']
            project.pending_job = job['id']
        with self.lock:
            self.active_tasks += 1
            self.jobs[job['id']] = job
            for old_id in list(self.jobs):
                if len(self.jobs) <= 200:
                    break
                if self.jobs[old_id]['state'] not in ('queued', 'running'):
                    del self.jobs[old_id]
        self.executor.submit(self._run_propagation, project, job, reference_masks, protected, snapshot)
        return copy.deepcopy(job)

    def _run_propagation(self, project, job, reference_masks, protected, snapshot):
        start = time.perf_counter()
        try:
            with self.lock:
                if job['state'] == 'cancelled':
                    return
                job['state'] = 'running'
                job['started_at'] = now()
            self._set_job_phase(job, 'preparing')
            z = job['slice']
            planes = {t: self._plane(project, t, z) for t in job['reference_times']}
            references = {label: [(planes[t], mask) for t, mask in items] for label, items in reference_masks.items()}
            masks, contexts, removed = {}, {}, 0

            def consume(entry, results, gpu_id=None):
                nonlocal removed
                labels, t = entry['organs'], entry['time']
                if set(results) != set(labels):
                    raise RuntimeError('In-context inference returned an incomplete set of structures.')
                for result in results.values():
                    p = result.get('probability')
                    if (p is None or p.shape != project.voxels.shape[2:] or not np.isfinite(p).all()
                            or np.any((p < 0) | (p > 1)) or result['mask'].shape != p.shape
                            or result['prior'].shape != (1, 512, 512) or not np.isfinite(result['prior']).all()):
                        raise RuntimeError('In-context inference returned invalid probabilities.')
                plane_masks, plane_contexts, count = self._exclusive_all(t, z, labels, results, protected[t])
                removed += count
                for label in labels:
                    key = (t, z, label)
                    masks[key] = plane_masks[label]
                    # Either click mode can refine the propagated prior. No
                    # reference clicks leak into target interaction history.
                    plane_contexts[key]['mode'] = None
                    plane_contexts[key]['propagation'] = {'reference_times': [time for time, _ in reference_masks[label]],
                                                         'slice': z, 'source_revision': job['base_revision']}
                contexts.update(plane_contexts)
                with self.lock:
                    job['completed_frames'] += 1
                    job['completed_slices'] += 1
                    if job['completed_slices'] == 1:
                        job['first_result_seconds'] = round(time.perf_counter() - start, 3)
                    job['current_time'] = t
                    if gpu_id is not None:
                        key = str(gpu_id)
                        job['gpu_completed'][key] = job['gpu_completed'].get(key, 0) + 1

            if job['gpu_ids'] is None:
                self._set_job_phase(job, 'predicting')
                for entry in job['preview']['eligible']:
                    with project.lock:
                        project.check_plane_unlocked(entry['time'], z)
                    image = self._plane(project, entry['time'], z)
                    results = {}
                    for label in entry['organs']:
                        with self.lock:
                            if job['state'] == 'cancelled':
                                return
                        results[label] = self.model.predict_reference(image, references[label], label)
                    consume(entry, results)
                status = self.model.status()
                checkpoint_sha = status['checkpoint_sha256']
                job['execution_devices'] = [status['device']]
            else:
                workers = self._initialize_gpu_workers(job,
                    'Propagation workers did not load the interactive model checkpoint.')
                if workers is None:
                    return
                checkpoint_sha = next(iter(workers.values())).checkpoint_sha256
                self._set_job_phase(job, 'predicting')
                entries = iter(job['preview']['eligible'])
                with ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix='kprism-propagation') as executor:
                    pending = {}

                    def dispatch(gpu_id):
                        with self.lock:
                            if job['state'] == 'cancelled':
                                return
                        entry = next(entries, None)
                        if entry is None:
                            return
                        with project.lock:
                            project.check_plane_unlocked(entry['time'], z)
                        image = self._plane(project, entry['time'], z)
                        task_id = f'{job["id"]}:{entry["time"]}:{z}'
                        future = executor.submit(workers[gpu_id].predict_reference, task_id, image,
                                                 entry['organs'], references)
                        pending[future] = (gpu_id, entry)

                    for gpu_id in workers:
                        dispatch(gpu_id)
                    while pending:
                        with self.lock:
                            if job['state'] == 'cancelled':
                                return
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            gpu_id, entry = pending.pop(future)
                            results = future.result()
                            with self.lock:
                                if job['state'] == 'cancelled':
                                    return
                            consume(entry, results, gpu_id)
                            dispatch(gpu_id)
            self._set_job_phase(job, 'applying')
            with project.lock, self.lock:
                if job['state'] == 'cancelled' or project.revision != job['base_revision'] or project.deleted:
                    self._finish_job(project, job, start, 'stale')
                    return
                if read_notes(project.directory)['review_status'] == 'complete':
                    raise ValueError('The study was marked complete. Reopen it before propagating.')
                self._check_job_frames(project, job)
                for (t, layer, label), mask in masks.items():
                    project.masks[label][t, layer] = mask
                project.contexts.update(contexts)
                changed_times = sorted({key[0] for key in masks})
                project.commit(snapshot, 'propagate_time', slice=z, changed_times=changed_times,
                    reference_times=job['reference_times'], organs=job['organs'],
                    skipped_locked_slices=job['skipped_locked_slices'], skipped_locked_frames=job['skipped_locked_frames'],
                    overwrite_edited=job['preview']['overwrite_edited'],
                    overlap_policy='highest_probability', background_threshold=.5, tie_break='ascending_label_id',
                    protected_voxels=removed, checkpoint=self.model.status()['checkpoint'],
                    checkpoint_sha256=checkpoint_sha, gpu_ids=job['gpu_ids'], execution_devices=job['execution_devices'])
                self._finish_job(project, job, start, 'completed', revision=project.revision,
                                 changed_times=changed_times, protected_voxels=removed,
                                 checkpoint_sha256=checkpoint_sha)
        except Exception as error:
            log.exception('Temporal propagation failed')
            self._finish_job(project, job, start, 'failed', error=str(error))
        finally:
            self._finish_job(project, job, start)
