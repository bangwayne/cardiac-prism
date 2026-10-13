"""Serialized real-model inference, feature reuse and revision-safe commits."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import re
import socket
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .domain import ORGANS, now
from .gpu_pool import GPUStartupCancelled, GPUWorkerPool
from .propagation import PropagationMixin
from . import cardiac_model
from .input_profile import prepared_source_hashes

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def padded_prior(mask, long_side=512):
    """Match the model preprocessor's nearest-neighbor resize and zero padding."""
    import torch
    import torch.nn.functional as functional
    height, width = mask.shape
    scale = long_side / max(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = functional.interpolate(tensor, size=(new_height, new_width), mode='nearest')
    top = (long_side - new_height) // 2
    left = (long_side - new_width) // 2
    padded = functional.pad(resized, (left, long_side - new_width - left,
                                     top, long_side - new_height - top))
    return padded[0].numpy().copy()


def manual_context(mask, previous_mode=None):
    return {'points': [], 'prior': padded_prior(mask), 'iteration': 1, 'mode': previous_mode}


def brush_mask(shape, points, radius):
    """Rasterize a whole stroke once in original image pixels."""
    result = np.zeros(shape, dtype=np.uint8)
    coordinates = np.floor(np.asarray(points) + .5).astype(np.int32)
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, shape[1] - 1)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, shape[0] - 1)
    for coordinate in coordinates:
        cv2.circle(result, tuple(coordinate), radius, 1, thickness=-1)
    if len(coordinates) > 1:
        cv2.polylines(result, [coordinates], isClosed=False, color=1, thickness=max(1, radius * 2))
    return result.astype(bool)


class ModelService:
    def __init__(self, predictor=None):
        self.api = None
        self.predictor = predictor
        self.error = None
        self.loading = False
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.checkpoint_sha256 = None
        self.profile = cardiac_model.model_profile() if predictor is None else {}
        self.cardiac_frames = OrderedDict()
        # Built-in legacy examples were already prepared offline. Do not infer
        # preprocessing from a numeric range, or percentile-clip them twice.
        self.prepared_example_hashes = prepared_source_hashes()
        self.requested_device = os.environ.get('CARDIAC_PRISM_DEVICE', 'cpu')
        self.features = OrderedDict()
        self.reference_features = OrderedDict()
        self.devices = None
        self.switching = False
        self.lock = threading.RLock()

    def available_devices(self):
        if self.devices is None:
            devices = [{'id': 'cpu', 'name': 'CPU'}]
            if self.predictor is None:
                from .cuda_devices import device_properties
                physical = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
                for index, properties in enumerate(device_properties()):
                    number = physical[index] if index < len(physical) and physical[index] else str(index)
                    devices.append({'id': f'cuda:{index}', 'name': f'GPU {number} · {properties.name}',
                                    'memory_gb': round(properties.total_memory / 1024**3, 1)})
            self.devices = devices
        return self.devices

    def set_device(self, requested):
        if not re.fullmatch(r'cpu|auto|cuda(?::\d+)?', requested):
            raise ValueError('Choose CPU or an available CUDA device.')
        available = {entry['id'] for entry in self.available_devices()}
        target = ('cuda:0' if 'cuda:0' in available else 'cpu') if requested == 'auto' else requested
        target = 'cuda:0' if target == 'cuda' else target
        if target not in available:
            raise ValueError('The requested GPU is unavailable on this server. Choose CPU or an available GPU.')
        with self.lock:
            self.switching = True
            try:
                if self.api is not None and str(self.api._model.device) != target:
                    import torch
                    self.features.clear()
                    self.reference_features.clear()
                    self.cache_stats = {'hits': 0, 'misses': 0}
                    # Release the previous GPU allocation before selecting another GPU.
                    self.api._model.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        self.api._model.to(target)
                    except Exception as error:
                        self.api._model.to('cpu')
                        torch.cuda.empty_cache()
                        self.requested_device = 'cpu'
                        self.api.requested_device = 'cpu'
                        raise ValueError(f'Could not move the model to {target}; restored CPU: {error}') from error
                    self.api.requested_device = target
                self.requested_device = target
                self.error = None
            finally:
                self.switching = False
        return self.status()

    def ensure_loaded(self):
        if self.api is not None:
            return
        self.loading = True
        try:
            import torch
            from cardiac_prism.api.segmentation_api import SegmentationAPI
            torch.set_num_threads(max(1, int(os.environ.get('CARDIAC_PRISM_CPU_THREADS', '2'))))
            api = SegmentationAPI(ROOT / 'cardiac_prism/config/eval_config.yaml', device=self.requested_device)
            with contextlib.redirect_stdout(io.StringIO()):
                api._ensure_loaded()
            original = api._model.backbone.forward
            features = self.features

            def cached(images):
                fingerprint = hashlib.sha256(images.detach().cpu().numpy().tobytes()).digest()
                key = (tuple(images.shape), str(images.device), str(images.dtype), fingerprint)
                if key in features:
                    features.move_to_end(key)
                    self.cache_stats['hits'] += 1
                    return features[key]
                self.cache_stats['misses'] += 1
                value = original(images)
                features[key] = value
                while len(features) > 4:
                    features.popitem(last=False)
                return value

            api._model.backbone.forward = cached
            mask_encoder = api._model.mask_encoder.forward

            def cached_reference(images):
                key = (str(images.device), str(images.dtype), tuple(images.shape),
                       hashlib.sha256(images.detach().cpu().numpy().tobytes()).digest())
                if key not in self.reference_features:
                    self.reference_features[key] = mask_encoder(images)
                self.reference_features.move_to_end(key)
                value = self.reference_features[key]
                while len(self.reference_features) > 1 and (len(self.reference_features) > 3 or
                        sum(v.numel() * v.element_size() for entry in self.reference_features.values()
                            for v in entry.values()) > 96 * 1024**2):
                    self.reference_features.popitem(last=False)
                return value

            api._model.mask_encoder.forward = cached_reference
            self.api = api
            checkpoint = Path(api.checkpoint_load_report['checkpoint_path'])
            with checkpoint.open('rb') as stream:
                fingerprint = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    fingerprint.update(chunk)
                self.checkpoint_sha256 = fingerprint.hexdigest()
            expected = self.profile.get('checkpoint_sha256')
            if expected and expected != self.checkpoint_sha256:
                self.api = None
                raise ValueError('Checkpoint checksum does not match its preprocessing profile.')
            self.error = None
            log.info('Cardiac-Prism model ready on %s', api._model.device)
        except Exception as error:
            self.error = str(error)
            raise
        finally:
            self.loading = False

    def predict(self, image, points, label, mode, prior, iteration):
        with self.lock:
            return self._predict(image, points, label, mode, prior, iteration)

    def predict_reference(self, image, references, label):
        with self.lock:
            if self.predictor is not None:
                return self.predictor.predict_reference(image, references, label)
            if self.profile.get('preprocessing') != cardiac_model.PROTOCOL:
                raise ValueError('Temporal propagation requires the cardiac model and its versioned preprocessing profile.')
            self.ensure_loaded()
            return cardiac_model.predict_reference(self.api, image, references, label)

    def prepare_plane(self, frame, z, cache_key, preprocessed=False, spacing=(1., 1., 1.)):
        if self.profile.get('preprocessing') != cardiac_model.PROTOCOL:
            return frame[z].copy()
        with self.lock:
            key = (cache_key, frame.shape, preprocessed, tuple(spacing))
            if key not in self.cardiac_frames:
                if preprocessed:
                    cardiac_model.validate_prepared(frame, dimensions=3)
                    self.cardiac_frames[key] = frame.astype(np.float32, copy=True)
                else:
                    self.cardiac_frames[key] = cardiac_model.prepare_frame(frame, spacing)
            self.cardiac_frames.move_to_end(key)
            prepared = self.cardiac_frames[key]
            while len(self.cardiac_frames) > 1 and (len(self.cardiac_frames) > 4 or
                    sum(v.nbytes for v in self.cardiac_frames.values()) > 64*1024**2):
                self.cardiac_frames.popitem(last=False)
            return cardiac_model.CardiacPlane(prepared[z].copy(), tuple(frame.shape[1:]))

    def mask_prior(self, mask):
        return cardiac_model.mask_prior(mask) if self.profile.get('preprocessing') == cardiac_model.PROTOCOL else padded_prior(mask)

    def _predict(self, image, points, label, mode, prior, iteration):
        if self.predictor is not None:
            return self.predictor(image, points, label, mode, prior, iteration)
        import torch
        self.ensure_loaded()
        if self.profile.get('preprocessing') == cardiac_model.PROTOCOL:
            if not isinstance(image, cardiac_model.CardiacPlane):
                # Standalone callers have only one slice; platform jobs always
                # pass a whole-frame-prepared plane to preserve training scaling.
                image = cardiac_model.CardiacPlane(cardiac_model.prepare_frame(image[None])[0], tuple(image.shape))
            return cardiac_model.predict(self.api, image, points, label, mode, prior, iteration)
        positive = [(point['x'], point['y']) for point in points if point['positive']]
        negative = [(point['x'], point['y']) for point in points if not point['positive']]
        result = self.api.segment_slice(
            image, positive, negative, None if mode == 'click' else label + 15,
            prev_mask=torch.from_numpy(prior.copy()) if prior is not None else None,
            click_index=iteration,
        )
        probability = result['seg_result_padded']
        if not torch.isfinite(probability).all():
            raise RuntimeError('Model produced non-finite probabilities.')
        return {'mask': result['mask'], 'probability': result['probability'], 'prior': probability.numpy().copy(),
                'iteration': result['click_index']}

    def status(self):
        return {
            'state': 'switching' if self.switching else 'ready' if self.api is not None else ('loading' if self.loading else 'idle'),
            'device': str(self.api._model.device) if self.api is not None else None,
            'requested_device': self.requested_device,
            'available_devices': self.available_devices(), 'hostname': socket.gethostname(),
            'error': self.error, 'feature_cache': self.cache_stats,
            'checkpoint': cardiac_model.checkpoint_path().name,
            'model_name': self.profile.get('name', 'Cardiac-Prism original'),
            'preprocessing': self.profile.get('preprocessing', 'legacy_ui_max_divide'),
            'source_checkpoint_sha256': self.profile.get('source_checkpoint_sha256'),
            'click_prompt_seed': 0,
            'checkpoint_sha256': self.checkpoint_sha256,
        }


class JobManager(PropagationMixin):
    def __init__(self, model):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kprism-inference')
        self.jobs = OrderedDict()
        self.lock = threading.RLock()
        self.active_tasks = 0
        self.gpu_pool = GPUWorkerPool()

    def _plane(self, project, t, z):
        prepared = project.input_profile.get('input_state') == 'trusted_prepared'
        return self.model.prepare_plane(project.voxels[t], z, (project.id, t), preprocessed=prepared,
                                        spacing=project.image.GetSpacing()[:3])

    def _set_job_phase(self, job, phase):
        with self.lock:
            if job['state'] != 'cancelled':
                job.update(phase=phase, phase_started_at=now())

    def _finish_job(self, project, job, started, state=None, **details):
        """Publish a terminal result only after releasing its study reservation.

        Commit paths call this while holding the same locks, so a cancellation
        cannot overtake an applied result. The finally path is intentionally
        idempotent. Cancellation may release the reservation earlier through
        the API; an old worker must never release a newer job's reservation.
        """
        with project.lock, self.lock:
            if 'finished_at' in job:
                return
            if project.pending_job == job['id']:
                project.pending_job = None
            self.active_tasks -= 1
            finished = now()
            if job['state'] == 'cancelled':
                state = 'cancelled'
                details.pop('error', None)
            elif state is None:
                state = 'failed'
                details.setdefault('error', 'Inference stopped before completion.')
            job.update(details)
            job.update(seconds=round(time.perf_counter() - started, 3), finished_at=finished)
            if 'phase' in job:
                job.update(phase=state, phase_started_at=finished)
            job['state'] = state

    def _initialize_gpu_workers(self, job, mismatch_message):
        """Load selected cards together, with cancellable, observable startup.

        Workers are reused by later jobs. All checkpoints must agree before
        dispatch, and all predictions remain private until the batch commits.
        """
        started = time.perf_counter()
        stop = threading.Event()
        self._set_job_phase(job, 'preparing')
        with self.lock:
            job.update(ready_gpu_ids=[], preparing_gpu_ids=[], gpu_errors={})

        def cancelled():
            with self.lock:
                return stop.is_set() or job['state'] == 'cancelled'

        def load(gpu_id):
            with self.lock:
                if cancelled():
                    raise GPUStartupCancelled('GPU startup was cancelled.')
                job['preparing_gpu_ids'].append(gpu_id)
            try:
                # Small test/custom pools can keep the original get(id) API.
                loader = getattr(self.gpu_pool, 'get_for_job', None)
                worker = loader(gpu_id, cancelled) if loader else self.gpu_pool.get(gpu_id)
                with self.lock:
                    if not cancelled():
                        job['ready_gpu_ids'].append(gpu_id)
                        job['ready_gpu_ids'].sort(key=job['gpu_ids'].index)
                        job['execution_devices'] = [f'GPU {value}' for value in job['ready_gpu_ids']]
                return worker
            except Exception as error:
                with self.lock:
                    if not isinstance(error, GPUStartupCancelled):
                        job['gpu_errors'][str(gpu_id)] = str(error)
                raise
            finally:
                with self.lock:
                    job['preparing_gpu_ids'].remove(gpu_id)

        workers = {}
        with ThreadPoolExecutor(max_workers=len(job['gpu_ids']), thread_name_prefix='kprism-gpu-startup') as executor:
            pending = {executor.submit(load, gpu_id): gpu_id for gpu_id in job['gpu_ids']}
            try:
                while pending:
                    if cancelled():
                        stop.set()
                        for future in pending:
                            future.cancel()
                        return None
                    done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        gpu_id = pending.pop(future)
                        try:
                            workers[gpu_id] = future.result()
                        except GPUStartupCancelled:
                            if cancelled():
                                return None
                            raise
            finally:
                # On failure/cancellation, already-started loads observe this
                # flag and release their process; queued loads cannot start.
                stop.set()
                for future in pending:
                    future.cancel()
        if job['state'] == 'cancelled':
            return None
        workers = {gpu_id: workers[gpu_id] for gpu_id in job['gpu_ids']}
        fingerprints = {worker.checkpoint_sha256 for worker in workers.values()}
        baseline = self.model.status()['checkpoint_sha256']
        if len(fingerprints) != 1 or baseline and baseline not in fingerprints:
            raise RuntimeError(mismatch_message)
        with self.lock:
            job['gpu_startup_seconds'] = round(time.perf_counter() - started, 3)
        return workers

    def set_device(self, requested):
        with self.lock:
            if self.active_tasks:
                raise ValueError('Wait for inference to finish before changing the model device, including cancelled tasks still computing.')
            return self.model.set_device(requested)

    def _check_job_frames(self, project, job):
        # Recheck at publication even though API lock changes are refused while
        # a job is active. No worker result may cross a protection boundary.
        if job.get('mode') == 'all_layers':
            planes = [(job['time'], z) for z in job['target_slices']]
        elif job.get('mode') in ('all_frames', 'propagation'):
            planes = job['target_slices']
        else:
            planes = [(job['time'], job['slice'])]
        for t, z in planes:
            project.check_plane_unlocked(t, z)

    def submit(self, project, request):
        with project.lock:
            project.check_revision(request['revision'])
            t, z, label = request['time'], request['slice'], request['organ']
            project.check_slice(t, z, label)
            project.check_plane_unlocked(t, z)
            if project.locks[label]:
                raise ValueError('Unlock this structure before editing it.')
            if project.pending_job:
                raise ValueError('This study already has an active inference task.')
            key = (t, z, label)
            snapshot = project.snapshot(key, 'Automatic segmentation' if request['mode'] == 'organ' else 'AI click')
            context = copy.deepcopy(project.context(key))
            mode = request['mode']
            if mode == 'organ':
                context = {'points': [], 'prior': None, 'iteration': 0, 'mode': mode}
            else:
                if self.model.profile and context.get('model_profile') != self.model.profile['id']:
                    # A saved/undone old prior can use a different image grid.
                    # Rebase just this organ/plane from its current annotation;
                    # native click coordinates and the saved study stay intact.
                    context['prior'] = self.model.mask_prior(project.masks[label][t, z])
                    context['iteration'] = 1
                    context.pop('mask_prior', None)
                # Manual edits have no prompts but deliberately retain a valid prior.
                if context['mode'] not in (None, mode, 'organ'):
                    context = {'points': [], 'prior': None, 'iteration': 0, 'mode': mode}
                if request.get('point') is None:
                    raise ValueError('Click mode requires a point.')
                point = request['point']
                if not (0 <= point['x'] < project.voxels.shape[3] and 0 <= point['y'] < project.voxels.shape[2]):
                    raise ValueError('Click is outside the source image.')
                if mode == 'click' and context['mode'] == 'organ':
                    context = {'points': [], 'prior': None, 'iteration': 0, 'mode': mode}
                if context.pop('mask_prior', False):
                    # Paired baseline annotations derive a prior from only the
                    # requested 2D plane, instead of allocating one per cine layer.
                    context['prior'] = self.model.mask_prior(project.masks[label][t, z])
                context['points'].append(point)
                if len(context['points']) > 100:
                    raise ValueError('A slice supports up to 100 clicks. Clear prompts to begin a new history.')
                context['mode'] = mode
            job_id = uuid.uuid4().hex
            job = {'id': job_id, 'project_id': project.id, 'time': t, 'slice': z, 'organ': label,
                   'state': 'queued', 'created_at': now(), 'seconds': 0,
                   'base_revision': project.revision, 'error': None}
            image = self._plane(project, t, z)
            project.pending_job = job_id
        with self.lock:
            self.active_tasks += 1
            self.jobs[job_id] = job
            # Keep active tasks, prune only completed job records.
            for old_id in list(self.jobs):
                if len(self.jobs) <= 200:
                    break
                if self.jobs[old_id]['state'] not in ('queued', 'running'):
                    del self.jobs[old_id]
        self.executor.submit(self._run, project, job, image, key, snapshot, context)
        return dict(job)

    def submit_all(self, project, request):
        with project.lock:
            project.check_revision(request['revision'])
            t, z = request['time'], request['slice']
            project.check_slice(t, z)
            project.check_plane_unlocked(t, z)
            if project.pending_job:
                raise ValueError('This study already has an active inference task.')
            labels = tuple(label for label in ORGANS if not project.locks[label])
            if not labels:
                raise ValueError('Unlock at least one structure before segmenting this slice.')
            locked_masks = [project.masks[label][t, z] for label in ORGANS if project.locks[label]]
            if locked_masks and np.any(sum(locked_masks) > 1):
                raise ValueError('Locked structures overlap on this slice. Unlock them before resolving labels.')
            snapshot = project.snapshot((t, z, labels), 'Segment all structures on this slice')
            job = {'id': uuid.uuid4().hex, 'project_id': project.id, 'time': t, 'slice': z, 'organ': 0,
                   'mode': 'all', 'organs': list(labels), 'state': 'queued', 'created_at': now(),
                   'seconds': 0, 'base_revision': project.revision, 'error': None}
            image = self._plane(project, t, z)
            project.pending_job = job['id']
        with self.lock:
            self.active_tasks += 1
            self.jobs[job['id']] = job
        self.executor.submit(self._run, project, job, image, (t, z, labels), snapshot, None)
        return dict(job)

    def submit_all_layers(self, project, request):
        with project.lock:
            project.check_revision(request['revision'])
            t = request['time']
            project.check_slice(t, 0)
            if project.pending_job:
                raise ValueError('This study already has an active inference task.')
            planes = project.unlocked_planes(times=[t])
            slices = [z for _, z in planes]
            labels = tuple(label for label in ORGANS if not project.locks[label])
            if not labels:
                raise ValueError('Unlock at least one structure before segmenting all layers.')
            locked_masks = [project.masks[label][t, slices] for label in ORGANS if project.locks[label]]
            if locked_masks and np.any(sum(locked_masks) > 1):
                raise ValueError('Locked structures overlap in this frame. Unlock them before resolving labels.')
            snapshot = project.snapshot_planes([(t, z, label) for z in slices for label in labels],
                                               'Segment all structures on all layers')
            protected = np.zeros(project.voxels.shape[1:], dtype=bool)
            for label in ORGANS:
                if project.locks[label]:
                    protected |= project.masks[label][t].astype(bool)
            job = {'id': uuid.uuid4().hex, 'project_id': project.id, 'time': t, 'slice': slices[0], 'organ': 0,
                   'mode': 'all_layers', 'organs': list(labels), 'state': 'queued', 'created_at': now(),
                   'seconds': 0, 'base_revision': project.revision, 'error': None,
                   'completed_slices': 0, 'total_slices': len(slices), 'current_slice': slices[0],
                   'target_slices': slices, 'skipped_locked_slices': [[frame, z] for frame, z in sorted(project.locked_slices)
                                                                    if frame == t],
                   'skipped_locked_frames': [t] if t in project.locked_frames else []}
            project.pending_job = job['id']
            # Images are immutable; only one plane is copied when its inference begins.
            images = project.voxels[t]
        with self.lock:
            self.active_tasks += 1
            self.jobs[job['id']] = job
            for old_id in list(self.jobs):
                if len(self.jobs) <= 200:
                    break
                if self.jobs[old_id]['state'] not in ('queued', 'running'):
                    del self.jobs[old_id]
        self.executor.submit(self._run_layers, project, job, images, labels, protected, snapshot)
        return dict(job)

    def _predict_all(self, image, labels, job):
        results = {}
        for label in labels:
            with self.lock:
                if job['state'] == 'cancelled':
                    return None
            # Each layer starts independently, without another layer's clicks or prior.
            results[label] = self.model.predict(image, [], label, 'organ', None, 0)
            probability = results[label].get('probability')
            if (probability is None or probability.shape != image.shape or
                    not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)) or
                    results[label]['mask'].shape != image.shape or
                    results[label]['prior'].shape != (1, 512, 512) or not np.isfinite(results[label]['prior']).all()):
                raise RuntimeError('Model output has an invalid probability shape or values.')
        return results

    def submit_all_frames(self, project, request):
        gpu_ids = request.get('gpu_ids')
        if gpu_ids is not None:
            self.gpu_pool.validate(gpu_ids)
        with project.lock:
            project.check_revision(request['revision'])
            if project.pending_job:
                raise ValueError('This study already has an active inference task.')
            labels = tuple(label for label in ORGANS if not project.locks[label])
            if not labels:
                raise ValueError('Unlock at least one structure before segmenting all time frames.')
            planes = project.unlocked_planes()
            times = sorted({t for t, _ in planes})
            locked = [label for label in ORGANS if project.locks[label]]
            if locked and any(np.any(sum(project.masks[label][t, z] for label in locked) > 1) for t, z in planes):
                raise ValueError('Locked structures overlap in this study. Unlock them before resolving labels.')
            frames, depth = project.voxels.shape[:2]
            snapshot = project.snapshot_planes([(t, z, label) for t, z in planes for label in labels],
                                               'Segment all structures on all time frames')
            protected = np.zeros(project.voxels.shape, dtype=bool)
            for label in ORGANS:
                if project.locks[label]:
                    protected[times] |= project.masks[label][times].astype(bool)
            job = {'id': uuid.uuid4().hex, 'project_id': project.id, 'time': times[0], 'slice': planes[0][1], 'organ': 0,
                   'mode': 'all_frames', 'organs': list(labels), 'state': 'queued', 'created_at': now(),
                   'seconds': 0, 'base_revision': project.revision, 'error': None,
                   'completed_slices': 0, 'total_slices': len(planes), 'current_slice': planes[0][1],
                   'current_time': times[0], 'completed_frames': 0, 'total_frames': len(times),
                   'target_times': times, 'skipped_locked_frames': sorted(project.locked_frames),
                   'target_slices': [list(pair) for pair in planes],
                   'skipped_locked_slices': [list(pair) for pair in sorted(project.locked_slices)],
                   'gpu_ids': gpu_ids, 'execution_devices': [], 'gpu_completed': {}}
            job.update(phase='queued', phase_started_at=job['created_at'], ready_gpu_ids=[],
                       preparing_gpu_ids=[], result_visibility='on_completion')
            project.pending_job = job['id']
            images = project.voxels
        with self.lock:
            self.active_tasks += 1; self.jobs[job['id']] = job
            for old_id in list(self.jobs):
                if len(self.jobs) <= 200:
                    break
                if self.jobs[old_id]['state'] not in ('queued', 'running'):
                    del self.jobs[old_id]
        self.executor.submit(self._run_frames, project, job, images, labels, protected, snapshot)
        with self.lock:
            return copy.deepcopy(job)

    def _run_frames(self, project, job, images, labels, protected, snapshot):
        started = time.perf_counter()
        try:
            with self.lock:
                if job['state'] == 'cancelled':
                    return
                job['state'] = 'running'
                job['started_at'] = now()
            self._set_job_phase(job, 'preparing')
            times = job['target_times']
            coordinates = [tuple(pair) for pair in job['target_slices']]
            masks = {label: {} for label in labels}
            contexts = {}; removed = 0; frames, depth = images.shape[:2]
            completed = {t: 0 for t in times}
            required = dict.fromkeys(times, 0)
            for t, _ in coordinates:
                required[t] += 1

            def consume(t, z, results, physical_id=None):
                nonlocal removed
                for result in results.values():
                    if (result['probability'].shape != images.shape[2:] or result['mask'].shape != images.shape[2:]
                            or result['prior'].shape != (1, 512, 512) or not np.isfinite(result['prior']).all()
                            or not np.isfinite(result['probability']).all() or np.any(result['probability'] < 0)
                            or np.any(result['probability'] > 1)):
                        raise RuntimeError('GPU worker returned invalid probability dimensions or values.')
                if set(results) != set(labels):
                    raise RuntimeError('GPU worker returned an incomplete set of structures.')
                plane_masks, plane_contexts, count = self._exclusive_all(t, z, labels, results, protected[t, z])
                for label in labels:
                    masks[label][(t, z)] = plane_masks[label]
                contexts.update(plane_contexts); removed += count; completed[t] += 1
                with self.lock:
                    job['completed_slices'] += 1
                    if job['completed_slices'] == 1:
                        job['first_result_seconds'] = round(time.perf_counter() - started, 3)
                    job['completed_frames'] = sum(completed[t] == required[t] for t in times)
                    job['current_time'] = t; job['current_slice'] = z
                    if physical_id is not None:
                        key = str(physical_id); job['gpu_completed'][key] = job['gpu_completed'].get(key, 0) + 1

            if job['gpu_ids'] is None:
                self._set_job_phase(job, 'predicting')
                for t, z in coordinates:
                    with project.lock:
                        project.check_plane_unlocked(t, z)
                    results = self._predict_all(self._plane(project, t, z), labels, job)
                    if results is None:
                        return
                    consume(t, z, results)
                status = self.model.status()
                execution_device = status['device']
                checkpoint_sha = status['checkpoint_sha256']
                with self.lock:
                    job['execution_devices'] = [execution_device]
            else:
                workers = self._initialize_gpu_workers(job,
                    'GPU workers did not load the same checkpoint as the interactive model.')
                if workers is None:
                    return
                checkpoint_sha = next(iter(workers.values())).checkpoint_sha256
                execution_device = ', '.join(f'GPU {gpu_id}' for gpu_id in workers)
                self._set_job_phase(job, 'predicting')
                dispatch_coordinates = iter(coordinates)
                # Only one plane is in flight per GPU. Cancellation stops new
                # dispatch, and draining keeps a cancelled result out of another job.
                with ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix='kprism-gpu-dispatch') as executor:
                    pending = {}
                    def dispatch(gpu_id):
                        with self.lock:
                            if job['state'] == 'cancelled':
                                return
                            coordinate = next(dispatch_coordinates, None)
                            if coordinate is None:
                                return
                            t, z = coordinate
                        with project.lock:
                            project.check_plane_unlocked(t, z)
                        task_id = f'{job["id"]}:{t}:{z}'
                        plane = self._plane(project, t, z)
                        future = executor.submit(workers[gpu_id].predict, task_id, plane, labels)
                        pending[future] = (gpu_id, t, z)
                    for gpu_id in workers:
                        dispatch(gpu_id)
                    while pending:
                        with self.lock:
                            if job['state'] == 'cancelled':
                                return
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            gpu_id, t, z = pending.pop(future)
                            results = future.result()
                            with self.lock:
                                if job['state'] == 'cancelled':
                                    return
                            consume(t, z, results, gpu_id); dispatch(gpu_id)
            self._set_job_phase(job, 'applying')
            with project.lock, self.lock:
                if job['state'] == 'cancelled' or project.revision != job['base_revision']:
                    self._finish_job(project, job, started, 'stale')
                    return
                self._check_job_frames(project, job)
                for label in labels:
                    for t, z in coordinates:
                        project.masks[label][t, z] = masks[label][(t, z)]
                project.contexts.update(contexts)
                status = self.model.status()
                project.commit(snapshot, 'segment_all_time_frames', frames=len(times), slices=len(coordinates),
                               changed_times=times, skipped_locked_frames=job['skipped_locked_frames'],
                               skipped_locked_slices=job['skipped_locked_slices'],
                               organs=list(labels), overlap_policy='highest_probability', background_threshold=0.5,
                               tie_break='ascending_label_id', protected_voxels=removed, device=execution_device,
                               checkpoint=status['checkpoint'], checkpoint_sha256=checkpoint_sha, click_prompt_seed=0,
                               gpu_ids=job['gpu_ids'])
                self._finish_job(project, job, started, 'completed', revision=project.revision,
                                 protected_voxels=removed, checkpoint_sha256=checkpoint_sha)
        except Exception as error:
            log.exception('All-time-frame inference failed')
            self._finish_job(project, job, started, 'failed', error=str(error))
        finally:
            self._finish_job(project, job, started)

    def _run_layers(self, project, job, images, labels, protected, snapshot):
        started = time.perf_counter()
        try:
            with self.lock:
                if job['state'] == 'cancelled':
                    return
                job['state'] = 'running'
            masks = {label: {} for label in labels}
            contexts = {}
            removed = 0
            for completed, z in enumerate(job['target_slices'], start=1):
                with self.lock:
                    if job['state'] == 'cancelled':
                        return
                    job['current_slice'] = z
                with project.lock:
                    project.check_plane_unlocked(job['time'], z)
                results = self._predict_all(self._plane(project, job['time'], z), labels, job)
                if results is None:
                    return
                plane_masks, plane_contexts, count = self._exclusive_all(
                    job['time'], z, labels, results, protected[z])
                for label in labels:
                    masks[label][z] = plane_masks[label]
                contexts.update(plane_contexts)
                removed += count
                with self.lock:
                    job['completed_slices'] = completed
            with project.lock, self.lock:
                if job['state'] == 'cancelled' or project.revision != job['base_revision']:
                    self._finish_job(project, job, started, 'stale')
                    return
                self._check_job_frames(project, job)
                status = self.model.status()
                for label in labels:
                    for z in job['target_slices']:
                        project.masks[label][job['time'], z] = masks[label][z]
                project.contexts.update(contexts)
                project.commit(snapshot, 'segment_all_layers', time_frame=job['time'],
                               slices=job['total_slices'], organs=list(labels),
                               changed_times=[job['time']], skipped_locked_slices=job['skipped_locked_slices'],
                               overlap_policy='highest_probability', background_threshold=0.5,
                               tie_break='ascending_label_id', protected_voxels=removed,
                               device=status['device'], checkpoint=status['checkpoint'],
                               checkpoint_sha256=status['checkpoint_sha256'], click_prompt_seed=status['click_prompt_seed'])
                self._finish_job(project, job, started, 'completed', revision=project.revision,
                                 protected_voxels=removed)
        except Exception as error:
            log.exception('All-layer inference task failed')
            self._finish_job(project, job, started, 'failed', error=str(error))
        finally:
            self._finish_job(project, job, started)

    def _run(self, project, job, image, key, snapshot, context):
        started = time.perf_counter()
        try:
            with self.lock:
                if job['state'] == 'cancelled':
                    return
                job['state'] = 'running'
            if context is None:
                results = self._predict_all(image, key[2], job)
                if results is None:
                    return
            else:
                result = self.model.predict(image, context['points'], key[2], context['mode'],
                                            context['prior'], context['iteration'])
            with project.lock, self.lock:
                if job['state'] == 'cancelled' or project.revision != job['base_revision']:
                    self._finish_job(project, job, started, 'stale')
                    return
                self._check_job_frames(project, job)
                if context is None:
                    self._commit_all(project, job, key, snapshot, results)
                    self._finish_job(project, job, started, 'completed')
                    return
                t, z, label = key
                mask = result['mask'].astype(np.uint8)
                if mask.shape != image.shape or not np.isfinite(result['prior']).all():
                    raise RuntimeError('Model output has an invalid shape or values.')
                protected = np.zeros_like(mask, dtype=bool)
                for other in ORGANS:
                    if other != label and project.locks[other]:
                        protected |= project.masks[other][t, z].astype(bool)
                removed = int((mask.astype(bool) & protected).sum())
                mask[protected] = 0
                changed = int(np.count_nonzero(project.masks[label][t, z] != mask))
                project.masks[label][t, z] = mask
                context['prior'] = self.model.mask_prior(mask) if removed else result['prior']
                context['iteration'] = result['iteration']
                if self.model.profile:
                    context['model_profile'] = self.model.profile['id']
                project.contexts[key] = context
                point = context['points'][-1] if context['mode'] != 'organ' and context['points'] else None
                project.commit(snapshot, 'inference', time_frame=t, slice=z, organ=label,
                               mode=context['mode'], points=context['points'],
                               point=point, click_polarity=('positive' if point['positive'] else 'negative') if point else None,
                               protected_voxels=removed, checkpoint=self.model.status()['checkpoint'],
                               checkpoint_sha256=self.model.status()['checkpoint_sha256'],
                               click_prompt_seed=self.model.status()['click_prompt_seed'])
                job['revision'] = project.revision
                job['protected_voxels'] = removed
                job['changed_voxels'] = changed
                if context['points']:
                    last = context['points'][-1]
                    x = min(mask.shape[1] - 1, max(0, int(round(last['x']))))
                    y = min(mask.shape[0] - 1, max(0, int(round(last['y']))))
                    job['point_foreground'] = bool(mask[y, x])
                self._finish_job(project, job, started, 'completed')
        except Exception as error:
            log.exception('Inference task failed')
            self._finish_job(project, job, started, 'failed', error=str(error))
        finally:
            self._finish_job(project, job, started)

    def _exclusive_all(self, t, z, labels, results, protected):
        probabilities = np.stack([results[label]['probability'] for label in labels])
        # Argmax on native image pixels. Ties use ascending label ID; <=0.5 remains background.
        winner = np.argmax(probabilities, axis=0)
        foreground = probabilities.max(axis=0) > 0.5
        removed = int((foreground & protected).sum())
        contexts = {}
        masks = {}
        for index, label in enumerate(labels):
            mask = ((winner == index) & foreground & ~protected).astype(np.uint8)
            masks[label] = mask
            # Preserve sigmoid confidence, while suppressing pixels assigned to another structure.
            prior = results[label]['prior'].copy()
            if np.any(results[label]['mask'].astype(bool) & ~mask.astype(bool)):
                prior *= self.model.mask_prior(mask)
            contexts[(t, z, label)] = {'points': [], 'mode': 'organ', 'iteration': results[label]['iteration'], 'prior': prior}
            if self.model.profile:
                contexts[(t, z, label)]['model_profile'] = self.model.profile['id']
        return masks, contexts, removed

    def _commit_all(self, project, job, key, snapshot, results):
        t, z, labels = key
        protected = np.zeros(project.voxels.shape[2:], dtype=bool)
        for label in ORGANS:
            if project.locks[label]:
                protected |= project.masks[label][t, z].astype(bool)
        masks, contexts, removed = self._exclusive_all(t, z, labels, results, protected)
        for label, mask in masks.items():
            project.masks[label][t, z] = mask
        project.contexts.update(contexts)
        status = self.model.status()
        project.commit(snapshot, 'segment_all', time_frame=t, slice=z, organs=list(labels),
                       overlap_policy='highest_probability', background_threshold=0.5, tie_break='ascending_label_id',
                       protected_voxels=removed, device=status['device'], checkpoint=status['checkpoint'],
                       checkpoint_sha256=status['checkpoint_sha256'], click_prompt_seed=status['click_prompt_seed'])
        job.update(revision=project.revision, protected_voxels=removed)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError('Unknown inference task.')
            return copy.deepcopy(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job['state'] in ('queued', 'running'):
                job['state'] = 'cancelled'
                if 'phase' in job:
                    job.update(phase='cancelled', phase_started_at=now())
            return copy.deepcopy(job)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.gpu_pool.close()
