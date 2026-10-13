"""Image geometry, independent structure masks, transactional edits and persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid
import zipfile

import numpy as np
import SimpleITK as sitk

from .state import PriorStore, load_state, save_state
from .study_notes import read_notes
from .annotation_progress import event_details
from .input_profile import build_input_profile

ORGANS = {1: {'name': 'Right ventricle', 'short': 'RV', 'color': '#f49a85'},
          2: {'name': 'Myocardium', 'short': 'MYO', 'color': '#52d9bd'},
          3: {'name': 'Left ventricle', 'short': 'LV', 'color': '#a697f4'}}
SCHEMA_VERSION = 2


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_locked_frames(times, frames):
    if (not isinstance(times, list) or any(type(t) is not int for t in times)
            or len(set(times)) != len(times)):
        raise ValueError('Locked time frames must be distinct integer frame indices.')
    if any(not 0 <= t < frames for t in times):
        raise ValueError('A locked time frame is outside the image.')
    return set(times)


def validate_locked_slices(pairs, frames, depth):
    if (not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2
            or any(type(value) is not int for value in pair) for pair in pairs)):
        raise ValueError('Locked slices must be distinct [time, layer] integer pairs.')
    result = {tuple(pair) for pair in pairs}
    if len(result) != len(pairs):
        raise ValueError('Locked slices must be distinct [time, layer] integer pairs.')
    if any(not 0 <= t < frames or not 0 <= z < depth for t, z in result):
        raise ValueError('A locked slice is outside the image.')
    return result


def locked_slices_from_manifest(manifest, frames, depth):
    legacy = validate_locked_frames(manifest.get('locked_frames', []), frames)
    if 'locked_slices' in manifest:
        return validate_locked_slices(manifest['locked_slices'], frames, depth)
    return {(t, z) for t in legacy for z in range(depth)}


def atomic_json(path: Path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def geometry(image):
    dimension = image.GetDimension()
    if dimension not in (3, 4) or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Use a scalar 3D or 4D NIfTI image.')
    matrix = np.array(image.GetDirection()).reshape(dimension, dimension)
    if dimension == 4 and (np.any(np.abs(matrix[3, :3]) > 1e-6) or np.any(np.abs(matrix[:3, 3]) > 1e-6)):
        raise ValueError('Coupled spatial and temporal directions are unsupported.')
    spatial = matrix[:3, :3]
    if not np.allclose(spatial.T @ spatial, np.eye(3), atol=1e-4):
        raise ValueError('Image direction must be an orthonormal spatial basis.')
    if min(image.GetSpacing()) <= 0:
        raise ValueError('Image spacing must be positive.')
    return {
        'dimension': dimension, 'size': list(image.GetSize()),
        'spacing': list(image.GetSpacing()), 'origin': list(image.GetOrigin()),
        'direction': list(image.GetDirection()),
    }


def array4d(image):
    array = sitk.GetArrayFromImage(image)
    return array[None] if array.ndim == 3 else array


def require_same_geometry(source, annotation):
    if source.GetSize() != annotation.GetSize() or source.GetDimension() != annotation.GetDimension():
        raise ValueError('Annotation dimensions differ from the source image.')
    for field in ('GetSpacing', 'GetOrigin', 'GetDirection'):
        if not np.allclose(getattr(source, field)(), getattr(annotation, field)(), rtol=0, atol=1e-5):
            raise ValueError(f'Annotation {field[3:].lower()} differs from the source image.')
    if annotation.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Annotation must be a scalar labelmap.')


@dataclass
class EditSnapshot:
    scope: tuple | None
    masks: dict
    contexts: dict
    description: str
    planes: tuple | None = None


@dataclass
class Project:
    id: str
    directory: Path
    name: str
    source_path: Path
    image: object
    voxels: np.ndarray
    reference: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    input_profile: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    contexts: dict = field(default_factory=dict)
    locks: dict = field(default_factory=lambda: {label: False for label in ORGANS})
    locked_slices: set = field(default_factory=set)
    revision: int = 0
    saved_revision: int = -1
    created_at: str = field(default_factory=now)
    modified_at: str = field(default_factory=now)
    events: list = field(default_factory=list)
    undo_stack: list = field(default_factory=list)
    redo_stack: list = field(default_factory=list)
    pending_job: str | None = None
    deleted: bool = False
    lock: object = field(default_factory=threading.RLock)
    save_lock: object = field(default_factory=threading.Lock)
    stats_cache: dict = field(default_factory=dict)
    dirty_layers: dict = field(default_factory=dict)
    reference_dirty_revision: int = -1
    last_save: dict = field(default_factory=dict)
    progress_cache: dict | None = None
    progress_mask_labels: object = None
    progress_dirty_planes: set = field(default_factory=set)
    progress_last_scanned: int = 0

    def __post_init__(self):
        self.prior_store = PriorStore(self.directory)
        self.source_geometry = geometry(self.image)
        if self.voxels.size > 128_000_000:
            raise ValueError('This local workstation supports up to 128 million voxels per study.')
        if not np.isfinite(self.voxels).all():
            raise ValueError('Image contains NaN or infinite values.')
        if not self.masks:
            self.masks = {label: np.zeros(self.voxels.shape, dtype=np.uint8) for label in ORGANS}
        sampled = self.voxels.reshape(-1)[::max(1, self.voxels.size // 250_000)]
        positive = sampled[sampled > 0]
        window_data = positive if positive.size else sampled
        low, high = np.percentile(window_data, [1, 99.5])
        self.display_window = [float(low), float(max(high, low + 1))]

    def check_slice(self, t, z, label=None):
        if self.deleted:
            raise KeyError('Unknown study.')
        if not 0 <= t < self.voxels.shape[0] or not 0 <= z < self.voxels.shape[1]:
            raise ValueError('Time frame or slice is outside the image.')
        if label is not None and label not in ORGANS:
            raise ValueError('Unknown structure.')

    def check_revision(self, revision):
        if self.deleted:
            raise KeyError('Unknown study.')
        if revision != self.revision:
            raise ValueError('The study changed. Refresh its state before applying this edit.')

    def check_frame_unlocked(self, t):
        self.check_slice(t, 0)
        if t in self.locked_frames:
            raise ValueError(f'Time frame {t + 1} is locked. Unlock this frame before changing its annotations.')

    @property
    def locked_frames(self):
        depth = self.voxels.shape[1]
        return {t for t in range(self.voxels.shape[0]) if all((t, z) in self.locked_slices for z in range(depth))}

    def check_plane_unlocked(self, t, z):
        self.check_slice(t, z)
        if (t, z) in self.locked_slices:
            raise ValueError(f'Time frame {t + 1}, layer {z + 1} is locked. Unlock this slice before changing its annotations.')

    def validate_locked_frames(self, times):
        return validate_locked_frames(times, self.voxels.shape[0])

    def unlocked_times(self):
        times = [t for t in range(self.voxels.shape[0]) if t not in self.locked_frames]
        if not times:
            raise ValueError('All time frames are locked. Unlock a frame before changing annotations.')
        return times

    def unlocked_planes(self, times=None, slices=None):
        times = range(self.voxels.shape[0]) if times is None else times
        slices = range(self.voxels.shape[1]) if slices is None else slices
        planes = [(t, z) for t in times for z in slices if (t, z) not in self.locked_slices]
        if not planes:
            raise ValueError('All selected slices in these time frames are locked. Unlock a slice before changing annotations.')
        return planes

    def set_frame_locks(self, revision, times, locked):
        self.set_slice_locks(revision, times, list(range(self.voxels.shape[1])), locked, frame_operation=True)

    def set_slice_locks(self, revision, times, slices, locked, frame_operation=False):
        """Publish protection atomically before exposing it to edits or jobs.

        Unsaved annotation edits remain unsaved. Their older manifest revision and
        events stay together; only the independently stored protection is updated.
        The save-lock/project-lock order matches normal saves. A failed write
        leaves every live field untouched, including a failed unlock request.
        """
        with self.save_lock, self.lock:
            self.check_revision(revision)
            if self.pending_job:
                raise ValueError('Wait for the active inference or cancel it before changing slice protection.')
            times = self.validate_locked_frames(times)
            slices = validate_locked_frames(slices, self.voxels.shape[1])
            if not times or not slices:
                raise ValueError('Select at least one time frame and layer.')
            targets = {(t, z) for t in times for z in slices}
            changed = sorted(targets - self.locked_slices if locked else targets & self.locked_slices)
            if not changed:
                return
            pairs = self.locked_slices | targets if locked else self.locked_slices - targets
            frames = {t for t in range(self.voxels.shape[0])
                      if all((t, z) in pairs for z in range(self.voxels.shape[1]))}
            next_revision, stamp = self.revision + 1, now()
            action = ('lock_frames' if locked else 'unlock_frames') if frame_operation else (
                     'lock_slices' if locked else 'unlock_slices')
            events = self.events + [{'time': stamp, 'action': action, 'revision': next_revision,
                'changed_times': sorted({t for t, _ in changed}), 'changed_slices': [list(pair) for pair in changed]}]
            path = self.directory / 'project.json'
            manifest = json.loads(path.read_text())
            manifest['locked_frames'] = sorted(frames)
            manifest['locked_slices'] = [list(pair) for pair in sorted(pairs)]
            clean = self.saved_revision == self.revision and manifest['revision'] == self.saved_revision
            if clean:
                manifest.update(revision=next_revision, modified_at=stamp, events=copy.deepcopy(events))
            atomic_json(path, manifest)
            # No other thread can observe an unlock before publication succeeds.
            self.locked_slices = pairs
            self.revision, self.modified_at, self.events = next_revision, stamp, events
            if clean:
                self.saved_revision = next_revision

    def snapshot_times(self, snapshot):
        if snapshot.planes is not None:
            return {key[0] for key in snapshot.planes}
        t = None if snapshot.scope is None else snapshot.scope[0]
        return set(range(self.voxels.shape[0])) if t is None else {t}

    def snapshot_pairs(self, snapshot):
        if snapshot.planes is not None:
            return {(t, z) for t, z, _ in snapshot.planes}
        t, z = (None, None) if snapshot.scope is None else snapshot.scope[:2]
        return {(frame, layer) for frame in (range(self.voxels.shape[0]) if t is None else (t,))
                for layer in (range(self.voxels.shape[1]) if z is None else (z,))}

    def context(self, key):
        if key not in self.contexts:
            self.contexts[key] = {'points': [], 'prior': None, 'iteration': 0, 'mode': None}
        return self.contexts[key]

    def snapshot(self, scope, description):
        if scope is None:
            masks = {label: mask.copy() for label, mask in self.masks.items()}
            contexts = copy.deepcopy(self.contexts)
        else:
            t, z, labels = scope
            labels = (labels,) if isinstance(labels, int) else labels
            if t is None:
                masks = {label: (self.masks[label] if z is None else self.masks[label][:, z]).copy()
                         for label in labels}
                contexts = {key: copy.deepcopy(value) for key, value in self.contexts.items()
                            if key[2] in labels and (z is None or key[1] == z)}
            elif z is None:
                # A frame edit spans slices, while preserving other frames and locked structures.
                masks = {label: self.masks[label][t].copy() for label in labels}
                contexts = {key: copy.deepcopy(value) for key, value in self.contexts.items()
                            if key[0] == t and key[2] in labels}
            else:
                masks = {label: self.masks[label][t, z].copy() for label in labels}
                contexts = {(t, z, label): copy.deepcopy(self.context((t, z, label))) for label in labels}
        return EditSnapshot(scope, masks, contexts, description)

    def apply_snapshot(self, snapshot):
        if self.snapshot_pairs(snapshot) & self.locked_slices:
            raise ValueError('Unlock the affected slices before restoring this edit.')
        if snapshot.planes is not None:
            for label, planes in snapshot.masks.items():
                for (t, z), mask in planes.items():
                    self.masks[label][t, z] = mask
            for key in snapshot.planes:
                self.contexts.pop(key, None)
            self.contexts.update(copy.deepcopy(snapshot.contexts))
            return
        if snapshot.scope is None:
            self.masks = {label: mask.copy() for label, mask in snapshot.masks.items()}
            self.contexts = copy.deepcopy(snapshot.contexts)
        else:
            t, z, _ = snapshot.scope
            for label, mask in snapshot.masks.items():
                if t is None:
                    if z is None:
                        self.masks[label] = mask.copy()
                    else:
                        self.masks[label][:, z] = mask
                elif z is None:
                    self.masks[label][t] = mask
                else:
                    self.masks[label][t, z] = mask
            if t is None:
                self.contexts = {key: value for key, value in self.contexts.items()
                                 if not (key[2] in snapshot.masks and (z is None or key[1] == z))}
            elif z is None:
                self.contexts = {key: value for key, value in self.contexts.items()
                                 if not (key[0] == t and key[2] in snapshot.masks)}
            self.contexts.update(copy.deepcopy(snapshot.contexts))

    def snapshot_planes(self, keys, description):
        """Freeze only affected organ/planes; saved priors remain lazy references."""
        keys = tuple(sorted(set(keys)))
        if not keys:
            raise ValueError('An edit must affect at least one plane.')
        masks = {}
        for t, z, label in keys:
            self.check_slice(t, z, label)
            masks.setdefault(label, {})[(t, z)] = self.masks[label][t, z].copy()
        depth = {key[1] for key in keys}
        scope = (None, next(iter(depth)) if len(depth) == 1 else None, tuple(sorted(masks)))
        return EditSnapshot(scope, masks,
            {key: copy.deepcopy(self.contexts[key]) for key in keys if key in self.contexts}, description, keys)

    def mark_snapshot_dirty(self, snapshot):
        if snapshot.planes is None:
            self.mark_dirty(snapshot.scope)
        else:
            for t, z, label in snapshot.planes:
                self.progress_dirty_planes.add((t, z))
                self.dirty_layers[(t, z, label)] = self.revision

    def commit(self, snapshot, action, **details):
        self.undo_stack.append(snapshot)
        # Bound memory used by full-volume imports and padded probability history.
        del self.undo_stack[:-40]
        self.redo_stack.clear()
        self.changed(action, **details, **event_details(self, snapshot, action, details))
        self.mark_snapshot_dirty(snapshot)

    def mark_dirty(self, scope):
        t, z, labels = scope if scope is not None else (None, None, tuple(ORGANS))
        labels = (labels,) if isinstance(labels, int) else labels
        frames = range(self.voxels.shape[0]) if t is None else (t,)
        slices = range(self.voxels.shape[1]) if z is None else (z,)
        for frame in frames:
            for layer in slices:
                self.progress_dirty_planes.add((frame, layer))
                for label in labels:
                    self.dirty_layers[(frame, layer, label)] = self.revision

    def changed(self, action, **details):
        self.revision += 1
        if action == 'import_reference':
            self.reference_dirty_revision = self.revision
        if action not in ('lock_frames', 'unlock_frames', 'lock_slices', 'unlock_slices'):
            frame = details.get('time_frame')
            if frame is None:
                self.stats_cache.clear()
            else:
                self.stats_cache.pop(frame, None)
        self.modified_at = now()
        self.events.append({'time': self.modified_at, 'action': action, 'revision': self.revision, **details})

    def undo(self, redo=False):
        if self.pending_job:
            raise ValueError('Wait for the active inference or cancel it before undoing.')
        source, destination = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not source:
            raise ValueError('No edit is available to redo.' if redo else 'No edit is available to undo.')
        snapshot = source[-1]
        if self.snapshot_pairs(snapshot) & self.locked_slices:
            raise ValueError('Unlock the affected slices before restoring this edit.')
        if snapshot.scope is None and any(self.locks.values()):
            raise ValueError('Unlock the structures before restoring a whole-study edit.')
        if snapshot.scope is not None and any(self.locks[label] for label in snapshot.masks):
            raise ValueError('Unlock this structure before restoring its edit.')
        source.pop()
        destination.append(self.snapshot_planes(snapshot.planes, snapshot.description) if snapshot.planes is not None
                           else self.snapshot(snapshot.scope, snapshot.description))
        self.apply_snapshot(snapshot)
        t, z, labels = snapshot.scope if snapshot.scope is not None else (None, None, tuple(ORGANS))
        labels = (labels,) if isinstance(labels, int) else labels
        self.changed('redo' if redo else 'undo', description=snapshot.description,
                     changed_times=sorted(self.snapshot_times(snapshot)),
                     changed_slices=[list(pair) for pair in sorted(self.snapshot_pairs(snapshot))],
                     annotation_scope=[t, z, list(labels)],
                     annotation_targets=[list(key) for key in snapshot.planes] if snapshot.planes is not None else
                                        [[frame, layer, label] for frame, layer in sorted(self.snapshot_pairs(snapshot))
                                         for label in labels])
        self.mark_snapshot_dirty(snapshot)

    def overlaps(self, t=None):
        arrays = [mask if t is None else mask[t] for mask in self.masks.values()]
        return int((sum(arrays) > 1).sum())

    def view(self, t=0, z=None):
        if z is None:
            z = self.voxels.shape[1] // 2
        self.check_slice(t, z)
        meta = self.source_geometry
        matrix = np.array(meta['direction']).reshape(meta['dimension'], meta['dimension'])[:3, :3]
        voxel_ml = float(np.prod(meta['spacing'][:3]) / 1000)
        if t not in self.stats_cache:
            self.stats_cache[t] = {
                'structures': {label: {'voxel_count': int(self.masks[label][t].sum()),
                    'annotated_slices': int(np.any(self.masks[label][t], axis=(1, 2)).sum())} for label in ORGANS},
                'overlaps': self.overlaps(t),
            }
        cached = self.stats_cache[t]
        structure_stats = []
        for label, definition in ORGANS.items():
            mask = self.masks[label][t]
            context = self.contexts.get((t, z, label), {'points': [], 'iteration': 0, 'mode': None})
            structure_stats.append({
                'id': label, **definition, 'locked': self.locks[label],
                **cached['structures'][label],
                'volume_ml': float(cached['structures'][label]['voxel_count'] * voxel_ml),
                'points': context['points'], 'iteration': context['iteration'], 'mode': context['mode'],
            })
        return {
            'id': self.id, 'name': self.name, 'shape': list(self.voxels.shape),
            'locked_frames': sorted(self.locked_frames),
            'locked_slices': [list(pair) for pair in sorted(self.locked_slices)],
            'provenance': self.provenance,
            'input_profile': self.input_profile,
            'dimensions': list(reversed(self.voxels.shape[1:])),
            'spacing': meta['spacing'][:3], 'origin': meta['origin'][:3],
            # vtk.js/gl-matrix store the image basis by columns; SimpleITK is row-major.
            'direction': matrix.T.reshape(-1).tolist(), 'source_geometry': meta,
            'display_window': self.display_window, 'structures': structure_stats,
            'revision': self.revision, 'saved_revision': self.saved_revision,
            'dirty': self.revision != self.saved_revision,
            'undo_count': len(self.undo_stack), 'redo_count': len(self.redo_stack),
            'has_reference': self.reference is not None, 'overlap_voxels': cached['overlaps'],
            'slice_overlap_voxels': int((sum(mask[t, z] for mask in self.masks.values()) > 1).sum()),
            'pending_job': self.pending_job, 'time': t, 'slice': z,
            'created_at': self.created_at, 'modified_at': self.modified_at,
            'events': self.events[-12:],
        }

    def save(self):
        save_state(self, SCHEMA_VERSION, atomic_json,
                   int(os.environ.get('CARDIAC_PRISM_SAVE_COMPRESSION_LEVEL', '1')))

    def nifti_bytes(self, array):
        scalar_array = array[0] if self.image.GetDimension() == 3 else array
        output = sitk.GetImageFromArray(scalar_array.astype(np.uint8), isVector=False)
        output.CopyInformation(self.image)
        temporary = self.directory / f'export-{uuid.uuid4().hex}.nii.gz'
        try:
            sitk.WriteImage(output, str(temporary))
            return temporary.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)

    def export(self, kind):
        if kind == 'bundle':
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for label, organ in ORGANS.items():
                    archive.writestr(f'{organ["short"]}.nii.gz', self.nifti_bytes(self.masks[label]))
                archive.writestr('annotation.json', json.dumps({
                    'study': self.name, 'geometry': self.source_geometry, 'revision': self.revision,
                    'labels': ORGANS, 'events': self.events, 'overlap_voxels': self.overlaps(),
                    'provenance': self.provenance,
                    'locked_frames': sorted(self.locked_frames),
                    'locked_slices': [list(pair) for pair in sorted(self.locked_slices)],
                    'study_notes': read_notes(self.directory),
                }, indent=2))
            return buffer.getvalue(), 'application/zip', 'structures.zip'
        if kind == 'labelmap':
            if self.overlaps():
                raise ValueError('Structure masks overlap. Resolve overlaps or export separate structure masks.')
            labels = np.zeros(self.voxels.shape, dtype=np.uint8)
            for label, mask in self.masks.items():
                labels[mask.astype(bool)] = label
            return self.nifti_bytes(labels), 'application/gzip', 'segmentation.nii.gz'
        try:
            label = int(kind)
        except ValueError:
            raise ValueError('Unknown export format.') from None
        if label not in ORGANS:
            raise ValueError('Unknown structure.')
        return self.nifti_bytes(self.masks[label]), 'application/gzip', f'{ORGANS[label]["short"]}.nii.gz'


class ProjectStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.projects = {}
        self.lock = threading.RLock()

    def create(self, path, name, reference=None):
        image = sitk.ReadImage(str(path))
        geometry(image)
        voxels = array4d(image).astype(np.float32)
        project_id = uuid.uuid4().hex
        directory = self.directory / project_id
        directory.mkdir()
        suffix = '.nii.gz' if str(path).endswith('.gz') else '.nii'
        source = directory / ('source' + suffix)
        try:
            shutil.copy2(path, source)
            project = Project(project_id, directory, name, source, image, voxels)
            project.input_profile = build_input_profile(source, voxels, project.source_geometry)
            if reference is not None:
                annotation = sitk.ReadImage(str(reference))
                require_same_geometry(image, annotation)
                values = array4d(annotation)
                if not np.isin(values, [0, 1, 2, 3]).all():
                    raise ValueError('Reference labels must be 0, 1 (RV), 2 (MYO), or 3 (LV).')
                project.reference = values.astype(np.uint8)
            project.save()
            with self.lock:
                self.projects[project_id] = project
            return project
        except Exception:
            shutil.rmtree(directory)
            raise

    def get(self, project_id):
        if not re.fullmatch(r'[a-f0-9]{32}', project_id):
            raise KeyError('Unknown study.')
        with self.lock:
            if project_id in self.projects:
                return self.projects[project_id]
            directory = self.directory / project_id
            manifest_path = directory / 'project.json'
            if not manifest_path.exists():
                raise KeyError('Unknown study.')
            manifest = json.loads(manifest_path.read_text())
            if manifest['schema_version'] not in (1, SCHEMA_VERSION):
                raise ValueError('Unsupported saved-project version.')
            for field in ('source_file', 'state_file'):
                if Path(manifest[field]).name != manifest[field]:
                    raise ValueError('Invalid project manifest path.')
            image = sitk.ReadImage(str(directory / manifest['source_file']))
            project = Project(project_id, directory, manifest['name'], directory / manifest['source_file'],
                              image, array4d(image).astype(np.float32))
            load_state(project, manifest)
            project.locked_slices = locked_slices_from_manifest(manifest, *project.voxels.shape[:2])
            project.locks = {int(key): value for key, value in manifest['locks'].items()}
            project.revision = project.saved_revision = manifest['revision']
            project.created_at = manifest['created_at']
            project.modified_at = manifest['modified_at']
            project.events = manifest['events']
            project.provenance = manifest.get('provenance', {})
            project.input_profile = build_input_profile(project.source_path, project.voxels, project.source_geometry)
            self.projects[project_id] = project
            return project

    def list(self):
        with self.lock:
            return self._list()

    def _list(self):
        records = []
        for path in self.directory.glob('*/project.json'):
            if not re.fullmatch(r'[a-f0-9]{32}', path.parent.name):
                continue
            data = json.loads(path.read_text())
            project = self.projects.get(data['id'])
            notes = read_notes(path.parent)
            records.append({'id': data['id'], 'name': data['name'],
                            'modified_at': project.modified_at if project else data['modified_at'],
                            'dirty': project.revision != project.saved_revision if project else False,
                            'review_status': notes['review_status'], 'note_preview': notes['text'][:160],
                            'notes_updated_at': notes['updated_at']})
        return sorted(records, key=lambda item: item['modified_at'], reverse=True)
