"""Immutable layer files and bounded, on-demand probability-prior reads."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import threading
import uuid
import weakref
import zipfile

import numpy as np


def file_path(directory, name):
    if not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('Invalid saved array file.')
    return directory / name


class LazyPrior:
    """An immutable stored prior; undo can retain its reference without reading it."""
    def __init__(self, store, filename, array, shape=(1, 512, 512), dtype='<f4'):
        file_path(store.directory, filename)
        self.store, self.filename, self.array = store, filename, array
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        if self.shape != (1, 512, 512) or self.dtype != np.dtype('float32'):
            raise ValueError('Saved prior shape or dtype is invalid.')
        store.references.add(self)

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __array__(self, dtype=None, copy=None):
        value = self.store.read(self)
        if dtype is not None and np.dtype(dtype) != value.dtype:
            return value.astype(dtype)
        return value.copy() if copy else value

    def copy(self):
        return self.store.read(self).copy()

    def __getitem__(self, index):
        return self.store.read(self)[index]

    def __deepcopy__(self, memo):
        return self


class PriorStore:
    def __init__(self, directory):
        self.directory = directory
        self.references = weakref.WeakSet()
        self.cache = OrderedDict()
        self.limit = 8 * 1024 * 1024
        self.reads = self.read_bytes = 0
        self.lock = threading.RLock()

    def reference(self, filename, array, shape=(1, 512, 512), dtype='<f4'):
        return LazyPrior(self, filename, array, shape, dtype)

    def read(self, prior):
        key = (prior.filename, prior.array)
        with self.lock:
            if key not in self.cache:
                with np.load(file_path(self.directory, prior.filename), allow_pickle=False) as archive:
                    value = archive[prior.array]
                if value.shape != prior.shape or value.dtype != prior.dtype or not np.isfinite(value).all():
                    raise ValueError('Saved probability prior is invalid.')
                value.flags.writeable = False
                self.cache[key] = value
                self.reads += 1
                self.read_bytes += value.nbytes
            self.cache.move_to_end(key)
            while sum(value.nbytes for value in self.cache.values()) > self.limit and len(self.cache) > 1:
                self.cache.popitem(last=False)
            return self.cache[key]

    def diagnostics(self):
        return dict(reads=self.reads, read_bytes=self.read_bytes,
                    cache_bytes=sum(value.nbytes for value in self.cache.values()))


def load_state(project, manifest):
    base = manifest['state_file']
    with np.load(file_path(project.directory, base), allow_pickle=False) as archive:
        project.masks = {label: archive[f'mask_{label}'].copy() for label in project.masks}
        project.reference = archive['reference'].copy() if 'reference' in archive else None
    grouped = {}
    for encoded, record in manifest.get('mask_layers', {}).items():
        key = tuple(map(int, encoded.split(':')))
        if len(key) != 3:
            raise ValueError('Invalid saved mask layer key.')
        project.check_slice(*key)
        grouped.setdefault(record['file'], []).append((key, record['array']))
    for filename, records in grouped.items():
        with np.load(file_path(project.directory, filename), allow_pickle=False) as archive:
            for (t, z, label), name in records:
                value = archive[name]
                if value.shape != project.voxels.shape[2:] or not np.isin(value, [0, 1]).all():
                    raise ValueError('Saved mask layer is invalid.')
                project.masks[label][t, z] = value
    if manifest.get('reference_file'):
        with np.load(file_path(project.directory, manifest['reference_file']), allow_pickle=False) as archive:
            project.reference = archive['reference'].copy()
    if any(mask.shape != project.voxels.shape or not np.isin(mask, [0, 1]).all()
           for mask in project.masks.values()):
        raise ValueError('Saved mask shape or values are invalid.')
    if project.reference is not None and (project.reference.shape != project.voxels.shape or
                                         not np.isin(project.reference, [0, 1, 2, 3]).all()):
        raise ValueError('Saved reference labelmap is invalid.')
    for record in manifest['contexts']:
        key = tuple(record['key'])
        project.check_slice(*key)
        context = {k: copy.deepcopy(v) for k, v in record.items()
                   if k not in ('key', 'prior_key', 'prior_file', 'prior_shape', 'prior_dtype')}
        context['prior'] = project.prior_store.reference(record.get('prior_file', base), record['prior_key'],
            record.get('prior_shape', (1, 512, 512)), record.get('prior_dtype', '<f4')) if 'prior_key' in record else None
        project.contexts[key] = context


def save_state(project, schema_version, atomic_json, compression_level):
    """Freeze only dirty layers; atomically publish references after writing them."""
    if not 0 <= compression_level <= 9:
        raise ValueError('CARDIAC_PRISM_SAVE_COMPRESSION_LEVEL must be between 0 and 9.')
    with project.save_lock:
        with project.lock:
            if project.deleted:
                raise KeyError('Unknown study.')
            manifest_path = project.directory / 'project.json'
            previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            if (project.saved_revision == project.revision and previous
                    and previous.get('provenance', {}) == project.provenance
                    and previous.get('input_profile', {}) == project.input_profile
                    and previous.get('locked_frames', []) == sorted(project.locked_frames)
                    and previous.get('locked_slices', []) == [list(pair) for pair in sorted(project.locked_slices)]):
                return
            revision = project.revision
            initial = previous is None
            filename = f'{"state" if initial else "layer"}-{revision}-{uuid.uuid4().hex[:8]}.npz'
            path = project.directory / filename
            arrays = {}
            mask_layers = copy.deepcopy(previous.get('mask_layers', {})) if previous else {}
            if initial:
                arrays.update({f'mask_{label}': mask.copy() for label, mask in project.masks.items()})
            else:
                for (t, z, label), changed_revision in project.dirty_layers.items():
                    if changed_revision <= revision:
                        name = f'mask_{t}_{z}_{label}'
                        arrays[name] = project.masks[label][t, z].copy()
                        mask_layers[f'{t}:{z}:{label}'] = dict(file=filename, array=name)
            reference_file = previous.get('reference_file') if previous else None
            if initial or project.reference_dirty_revision > project.saved_revision:
                if project.reference is not None:
                    arrays['reference'] = project.reference.copy()
                    reference_file = filename if not initial else None
            contexts, originals = [], {}
            for key, context in project.contexts.items():
                record = copy.deepcopy({k: v for k, v in context.items() if k != 'prior'})
                record['key'] = list(key)
                prior = context['prior']
                if isinstance(prior, LazyPrior):
                    record.update(prior_file=prior.filename, prior_key=prior.array,
                                  prior_shape=list(prior.shape), prior_dtype=prior.dtype.str)
                elif prior is not None:
                    # NumPy priors are new model/manual results; saved ones are lazy references.
                    name = f'prior_{key[0]}_{key[1]}_{key[2]}'
                    arrays[name] = prior.copy()
                    originals[key] = prior
                    record.update(prior_file=filename, prior_key=name,
                                  prior_shape=list(prior.shape), prior_dtype=prior.dtype.str)
                contexts.append(record)
            manifest = dict(schema_version=schema_version, storage='layer_chunks',
                id=project.id, name=project.name, source_file=project.source_path.name,
                state_file=filename if initial else previous['state_file'], mask_layers=mask_layers,
                reference_file=reference_file, revision=revision, locks=dict(project.locks), contexts=contexts,
                locked_frames=sorted(project.locked_frames),
                locked_slices=[list(pair) for pair in sorted(project.locked_slices)],
                created_at=project.created_at, modified_at=project.modified_at,
                events=copy.deepcopy(project.events), source_geometry=project.source_geometry,
                provenance=copy.deepcopy(project.provenance), input_profile=copy.deepcopy(project.input_profile))
        try:
            if arrays:
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=compression_level) as archive:
                    for name, array in arrays.items():
                        with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                            np.lib.format.write_array(stream, array, allow_pickle=False)
            atomic_json(manifest_path, manifest)
            with project.lock:
                project.saved_revision = revision
                project.dirty_layers = {key: value for key, value in project.dirty_layers.items() if value > revision}
                if project.reference_dirty_revision <= revision:
                    project.reference_dirty_revision = -1
                for record in contexts:
                    key = tuple(record['key'])
                    if key in originals and project.contexts.get(key, {}).get('prior') is originals[key]:
                        project.contexts[key]['prior'] = project.prior_store.reference(filename,
                            record['prior_key'], record['prior_shape'], record['prior_dtype'])
                protected = {manifest['state_file']}
                protected.update(item['file'] for item in mask_layers.values())
                protected.update(record['prior_file'] for record in contexts if 'prior_file' in record)
                protected.update(prior.filename for prior in project.prior_store.references)
                if reference_file:
                    protected.add(reference_file)
            for pattern in ('layer-*.npz', 'state-*.npz'):
                for candidate in project.directory.glob(pattern):
                    if candidate.name not in protected:
                        candidate.unlink(missing_ok=True)
            project.last_save = dict(revision=revision, arrays=list(arrays),
                copied_bytes=sum(array.nbytes for array in arrays.values()),
                written_bytes=path.stat().st_size if arrays else 0,
                **project.prior_store.diagnostics())
        except Exception:
            if not manifest_path.exists() or filename not in json.dumps(json.loads(manifest_path.read_text())):
                path.unlink(missing_ok=True)
            raise
