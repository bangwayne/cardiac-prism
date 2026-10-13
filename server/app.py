"""Local imaging API and same-origin production frontend."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Literal, Annotated

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, FileResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from .domain import ORGANS, ProjectStore, array4d, require_same_geometry
from .inference import JobManager, ModelService, brush_mask, manual_context
from .gpu_pool import gpu_inventory
from .paired_annotations import catalog, initialize_masks, read_annotation
from .study_notes import StudyNotesStore
from .library_state import LibraryState
from .study_backup import create_backup, import_backup
from .study_management import set_completion, delete_study, cleanup_deleted
from .annotation_progress import summary as annotation_summary
from .activity_history import activity
from .dicom_import import import_folder, ConverterUnavailable

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def demo_provenance():
    """Optional catalogue metadata; installations may start without examples."""
    path = ROOT / 'demo-data/provenance.json'
    if not path.exists():
        return {}
    return {entry['name']: entry for entry in json.loads(path.read_text())}


class RevisionRequest(BaseModel):
    revision: int = Field(ge=0)


class Point(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    x: float
    y: float
    positive: bool


class InferenceRequest(RevisionRequest):
    time: int = Field(ge=0)
    slice: int = Field(ge=0)
    organ: int = Field(ge=1, le=3)
    mode: Literal['organ', 'click', 'organ_click']
    point: Point | None = None


class SliceRequest(RevisionRequest):
    time: int = Field(ge=0)
    slice: int = Field(ge=0)
    organ: int = Field(ge=1, le=3)


class SegmentAllRequest(RevisionRequest):
    time: int = Field(ge=0)
    slice: int = Field(ge=0)


class AllLayersRequest(RevisionRequest):
    time: int = Field(ge=0)


class AllFramesRequest(RevisionRequest):
    gpu_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = Field(default=None, max_length=16)


class PropagationRequest(AllFramesRequest):
    slice: int = Field(strict=True, ge=0)
    reference_times: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1, max_length=3)
    organs: list[Annotated[int, Field(strict=True, ge=1, le=3)]] = Field(min_length=1, max_length=3)
    target_times: list[Annotated[int, Field(strict=True, ge=0)]] | None = Field(default=None, min_length=1)
    overwrite_edited: bool = False


class DeviceRequest(BaseModel):
    device: str = Field(max_length=20)


class BrushRequest(SliceRequest):
    points: list[tuple[float, float]] = Field(min_length=1, max_length=2000)
    radius: int = Field(ge=0, le=30)
    erase: bool = False


class LockRequest(RevisionRequest):
    organ: int = Field(ge=1, le=3)
    locked: bool


class FrameLockRequest(RevisionRequest):
    times: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1)
    locked: bool


class SliceLockRequest(FrameLockRequest):
    slices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1)


class ResolveRequest(RevisionRequest):
    time: int = Field(ge=0)
    slice: int = Field(ge=0)
    priority: int = Field(ge=1, le=3)


class DemoRequest(BaseModel):
    name: str


class StudyNotesRequest(RevisionRequest):
    text: str = Field(max_length=5000)
    review_status: Literal['in_progress', 'complete']


class LibraryVisibilityRequest(BaseModel):
    name: str
    hidden: bool


class StudyManagementRequest(RevisionRequest):
    notes_revision: int = Field(ge=0)


class StudyCompletionRequest(StudyManagementRequest):
    review_status: Literal['in_progress', 'complete']


def create_app(workspace=None, model=None):
    directory = Path(workspace or os.environ.get('CARDIAC_PRISM_WORKSPACE', ROOT / 'workspace'))
    store = ProjectStore(directory / 'projects')
    notes = StudyNotesStore(store.directory)
    library = LibraryState(directory)
    service = model or ModelService()
    manager = JobManager(service)

    @asynccontextmanager
    async def lifespan(application):
        if os.environ.get('CARDIAC_PRISM_PRELOAD') == '1':
            service.ensure_loaded()
        yield
        manager.close()
        for project in store.projects.values():
            if project.revision != project.saved_revision:
                try:
                    project.save()
                except OSError:
                    log.exception('Could not save study %s during shutdown', project.id)

    application = FastAPI(title='Cardiac-Prism Workstation', version='0.1.1', lifespan=lifespan)
    application.state.store = store
    application.state.model = service
    application.state.jobs = manager
    gzip_level = int(os.environ.get('CARDIAC_PRISM_GZIP_LEVEL', '1'))
    if not 0 <= gzip_level <= 9:
        raise ValueError('CARDIAC_PRISM_GZIP_LEVEL must be between 0 and 9.')
    # Float32 cine frames spend several seconds at the previous level 9.
    # Level 1 remains lossless and leaves CPU time for interactive rendering.
    application.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=gzip_level)

    @application.middleware('http')
    async def browser_headers(request, call_next):
        response = await call_next(request)
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith('/api') else 'no-cache'
        return response

    @application.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({'detail': str(error)}, status_code=400)

    @application.exception_handler(KeyError)
    async def key_error(request, error):
        return JSONResponse({'detail': str(error).strip("'")}, status_code=404)

    def editable(project, revision, organ=None, time=None, slice=None):
        project.check_revision(revision)
        if project.pending_job:
            raise ValueError('Wait for the active inference or cancel it before editing.')
        if organ is not None and project.locks[organ]:
            raise ValueError('Unlock this structure before editing it.')
        if time is not None:
            project.check_plane_unlocked(time, slice)

    def upload_path(file, backup=False):
        filename = Path(file.filename or '').name
        if not (filename.endswith('.zip') if backup else filename.endswith('.nii') or filename.endswith('.nii.gz')):
            raise ValueError('Choose a .nii or .nii.gz file.')
        upload_dir = directory / 'uploads'
        upload_dir.mkdir(exist_ok=True, parents=True)
        descriptor, name = tempfile.mkstemp(dir=upload_dir, suffix='.zip' if backup else '.nii.gz' if filename.endswith('.gz') else '.nii')
        size = 0
        try:
            with os.fdopen(descriptor, 'wb') as output:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > (2 * 1024 ** 3 if backup else 512 * 1024 * 1024):
                        raise ValueError('The upload limit is 2 GB for study backups and 512 MB for NIfTI files.')
                    output.write(chunk)
            if not size:
                raise ValueError('The uploaded file is empty.')
        except Exception:
            Path(name).unlink(missing_ok=True)
            raise
        return Path(name), filename

    @application.get('/api/health')
    def health():
        return {'status': 'ok', 'model': service.status(), 'version': '0.1.1',
                'features': {'study_notes': True, 'study_lifecycle': True, 'study_management': True,
                             'annotation_progress': True, 'temporal_propagation': True, 'frame_locks': True,
                             'slice_locks': True, 'activity_history': True, 'dicom_import': True},
                'inference_plane': 'original acquisition slices', 'workspace': str(directory)}

    @application.get('/api/demos')
    def demos(include_hidden: bool = False):
        paired = catalog(ROOT)
        provenance = demo_provenance()
        state = library.state()
        hidden, deleted = set(state.get('hidden_examples', [])), set(state.get('deleted_examples', []))
        return [{'name': path.name, 'patient': provenance.get(path.name, {}).get('display_name', path.name.split('_')[0]),
                 'phase': provenance.get(path.name, {}).get('phase', 'NIfTI volume'),
                 **{key: provenance[path.name][key] for key in ('time_frames', 'slices')
                    if key in provenance.get(path.name, {})},
                 'has_reference': (ROOT / 'demo-data/references' / (path.name[:-7] + '_gt.nii.gz')).exists(),
                 'has_annotation': path.name in paired,
                 'annotation_name': paired.get(path.name, {}).get('filename'), 'hidden': path.name in hidden}
                for path in sorted((ROOT / 'demo-data/images').glob('*.nii.gz'))
                if path.name not in deleted and (include_hidden or path.name not in hidden)]

    @application.post('/api/demos/visibility')
    def demo_visibility(request: LibraryVisibilityRequest):
        if request.name not in {item['name'] for item in demos(True)}:
            raise ValueError('Unknown demo scan.')
        library.set_hidden(request.name, request.hidden)
        return {'name': request.name, 'hidden': request.hidden}

    @application.post('/api/demos/delete')
    def delete_example(request: DemoRequest):
        if request.name not in {item['name'] for item in demos(True)}:
            raise ValueError('Unknown demo scan.')
        library.remove_permanently(request.name)
        return {'name': request.name, 'deleted': True}

    @application.get('/api/gpus')
    def physical_gpus():
        return gpu_inventory()

    @application.post('/api/model/device')
    def model_device(request: DeviceRequest):
        return manager.set_device(request.device)

    @application.post('/api/demos/open')
    def open_demo(request: DemoRequest):
        names = {item['name'] for item in demos()}
        if request.name not in names:
            raise ValueError('Unknown demo scan.')
        reference = ROOT / 'demo-data/references' / (request.name[:-7] + '_gt.nii.gz')
        project = store.create(ROOT / 'demo-data/images' / request.name, request.name,
                               reference if reference.exists() else None)
        project.provenance = demo_provenance().get(request.name, {})
        if request.name in catalog(ROOT):
            try:
                entry, labels = read_annotation(ROOT, project.image, request.name)
                initialize_masks(project, labels)
                project.provenance = {**project.provenance, 'initial_annotation': entry['filename'],
                                      'initial_annotation_sha256': entry['sha256']}
                project.changed('initialize_paired_annotation', filename=entry['filename'], sha256=entry['sha256'])
                project.mark_dirty(None)
            except Exception:
                # A failed pairing must never leave a misleading empty library study.
                with store.lock:
                    store.projects.pop(project.id, None)
                import shutil
                shutil.rmtree(project.directory)
                raise
        project.save()
        return project.view()

    @application.get('/api/projects')
    def projects():
        return store.list()

    @application.post('/api/projects')
    def upload_study(file: UploadFile = File(...)):
        path, name = upload_path(file)
        try:
            try:
                return store.create(path, name).view()
            except RuntimeError as error:
                raise ValueError('The file could not be read as a NIfTI image.') from error
        finally:
            path.unlink(missing_ok=True)

    @application.post('/api/projects/import-backup')
    def restore_backup(file: UploadFile = File(...)):
        path, _ = upload_path(file, backup=True)
        try:
            return import_backup(store, path).view()
        finally:
            path.unlink(missing_ok=True)

    @application.post('/api/projects/import-dicom')
    def upload_dicom(files: list[UploadFile] = File(...), folder_name: str | None = Form(default=None)):
        try:
            return import_folder(store, files, folder_name).view()
        except ConverterUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get('/api/projects/{project_id}')
    def project_view(project_id: str, time: int = 0, slice: int | None = None):
        project = store.get(project_id)
        with project.lock:
            return project.view(time, slice)

    @application.get('/api/projects/{project_id}/progress')
    def study_progress(project_id: str):
        project = store.get(project_id)
        with project.lock:
            return annotation_summary(project)

    @application.get('/api/projects/{project_id}/activity')
    def study_activity(project_id: str, before_revision: int | None = None, limit: int = 50,
                       time: int | None = None, slice: int | None = None):
        project = store.get(project_id)
        with project.lock:
            return activity(project, before_revision=before_revision, limit=limit, time=time, slice=slice)

    @application.get('/api/projects/{project_id}/notes')
    def study_notes(project_id: str):
        return notes.get(project_id)

    @application.post('/api/projects/{project_id}/notes')
    def update_study_notes(project_id: str, request: StudyNotesRequest):
        return notes.save(project_id, **request.model_dump())

    @application.post('/api/projects/{project_id}/completion')
    def study_completion(project_id: str, request: StudyCompletionRequest):
        return set_completion(store.get(project_id), notes, **request.model_dump())

    @application.post('/api/projects/{project_id}/delete')
    def remove_study(project_id: str, request: StudyManagementRequest):
        target = delete_study(store, notes, store.get(project_id), **request.model_dump())
        return JSONResponse({'id': project_id, 'deleted': True}, background=BackgroundTask(cleanup_deleted, target))

    @application.get('/api/projects/{project_id}/volume')
    def volume(project_id: str, time: int = 0):
        project = store.get(project_id)
        project.check_slice(time, 0)
        return Response(project.voxels[time].astype('<f4', copy=False).tobytes(),
                        media_type='application/octet-stream', headers={'X-Array-Dtype': 'float32'})

    @application.get('/api/projects/{project_id}/masks')
    def masks(project_id: str, time: int = 0, slice: int | None = None):
        project = store.get(project_id)
        with project.lock:
            project.check_slice(time, 0 if slice is None else slice)
            packed = np.stack([project.masks[label][time] if slice is None else project.masks[label][time, slice]
                               for label in ORGANS])
            headers = {'X-Array-Dtype': 'uint8', 'X-Array-Channels': 'RV,MYO,LV',
                       'X-Project-Revision': str(project.revision)}
            if slice is not None:
                headers['X-Mask-Slice'] = str(slice)
            return Response(packed.tobytes(), media_type='application/octet-stream',
                            headers=headers)

    @application.get('/api/projects/{project_id}/reference')
    def reference(project_id: str, time: int = 0):
        project = store.get(project_id)
        project.check_slice(time, 0)
        if project.reference is None:
            raise KeyError('No reference labelmap is available.')
        return Response(project.reference[time].tobytes(), media_type='application/octet-stream')

    @application.post('/api/projects/{project_id}/inference')
    def inference(project_id: str, request: InferenceRequest):
        return manager.submit(store.get(project_id), request.model_dump())

    @application.post('/api/projects/{project_id}/segment-all')
    def segment_all(project_id: str, request: SegmentAllRequest):
        return manager.submit_all(store.get(project_id), request.model_dump())

    @application.post('/api/projects/{project_id}/segment-all-layers')
    def segment_all_layers(project_id: str, request: AllLayersRequest):
        return manager.submit_all_layers(store.get(project_id), request.model_dump())

    @application.post('/api/projects/{project_id}/segment-all-time-frames')
    def segment_all_time_frames(project_id: str, request: AllFramesRequest):
        return manager.submit_all_frames(store.get(project_id), request.model_dump())

    @application.post('/api/projects/{project_id}/propagation-preview')
    def propagation_preview(project_id: str, request: PropagationRequest):
        return manager.propagation_preview(store.get(project_id), request.model_dump())

    @application.post('/api/projects/{project_id}/propagate-time')
    def propagate_time(project_id: str, request: PropagationRequest):
        return manager.submit_propagation(store.get(project_id), request.model_dump())

    @application.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return manager.get(job_id)

    @application.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        result = manager.cancel(job_id)
        project = store.get(result['project_id'])
        with project.lock:
            if project.pending_job == job_id:
                project.pending_job = None
        return result

    @application.post('/api/projects/{project_id}/brush')
    def brush(project_id: str, request: BrushRequest):
        project = store.get(project_id)
        with project.lock:
            editable(project, request.revision, request.organ, request.time, request.slice)
            project.check_slice(request.time, request.slice, request.organ)
            coordinates = np.asarray(request.points)
            if not np.isfinite(coordinates).all():
                raise ValueError('Brush coordinates must be finite.')
            height, width = project.voxels.shape[2:]
            if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= width) or np.any(coordinates[:, 1] >= height):
                raise ValueError('Brush coordinates are outside the image.')
            key = (request.time, request.slice, request.organ)
            painted = brush_mask((height, width), coordinates, request.radius)
            mask = project.masks[request.organ][request.time, request.slice]
            affected = [request.organ]
            if not request.erase:
                for label in ORGANS:
                    if label != request.organ and project.locks[label]:
                        painted &= ~project.masks[label][request.time, request.slice].astype(bool)
                affected += [label for label in ORGANS if label != request.organ and not project.locks[label]
                             and np.any(project.masks[label][request.time, request.slice][painted])]
            snapshot = project.snapshot((request.time, request.slice, tuple(affected)),
                                        'Erase stroke' if request.erase else 'Brush stroke')
            if request.erase:
                mask[painted] = 0
            else:
                mask[painted] = 1
                for label in affected[1:]:
                    project.masks[label][request.time, request.slice][painted] = 0
            for label in affected:
                context_key = (request.time, request.slice, label)
                project.contexts[context_key] = manual_context(project.masks[label][request.time, request.slice],
                                                              project.context(context_key)['mode'])
            project.commit(snapshot, 'erase' if request.erase else 'brush',
                           time_frame=request.time, slice=request.slice, organ=request.organ,
                           radius=request.radius, affected_organs=affected, overlap_policy='painted_structure_wins')
            return project.view(request.time, request.slice)

    @application.post('/api/projects/{project_id}/demo-predictions')
    def load_demo_predictions(project_id: str, request: RevisionRequest):
        project = store.get(project_id)
        name = project.provenance.get('name')
        if name not in {item['name'] for item in demos()}:
            raise ValueError('Evaluated example masks are available only for bundled demo studies.')
        directory = ROOT / 'demo-data/predictions'
        evaluation = json.loads((directory / 'evaluation.json').read_text())
        case = next(item for item in evaluation['cases'] if item['file'] == name)
        predictions = {}
        for label, organ in ORGANS.items():
            annotation = sitk.ReadImage(str(directory / (name[:-7] + f'_{organ["short"]}_single_click.nii.gz')))
            require_same_geometry(project.image, annotation)
            predictions[label] = array4d(annotation).astype(np.uint8)
        with project.lock:
            editable(project, request.revision)
            if any(project.locks.values()):
                raise ValueError('Unlock all structures before loading the evaluated example.')
            planes = project.unlocked_planes()
            times = sorted({t for t, _ in planes})
            keys = [(t, z, label) for t, z in planes for label in ORGANS]
            snapshot = project.snapshot_planes(keys, 'Load evaluated AI example')
            for t, z, label in keys:
                project.masks[label][t, z] = predictions[label][t, z]
                project.contexts.pop((t, z, label), None)
            for label in ORGANS:
                for t, z in np.argwhere(np.any(predictions[label], axis=(2, 3))):
                    if (int(t), int(z)) not in project.locked_slices:
                        project.contexts[(int(t), int(z), label)] = manual_context(predictions[label][t, z])
            project.commit(snapshot, 'import_evaluated_predictions', protocol=evaluation['protocol'],
                           changed_times=times, skipped_locked_frames=sorted(project.locked_frames),
                           skipped_locked_slices=[list(pair) for pair in sorted(project.locked_slices)],
                           source_report=evaluation['source_report'],
                           dice={short: info['single_click_3d_dice'] for short, info in case['organs'].items()})
        project.save()
        with project.lock:
            return project.view()

    @application.post('/api/projects/{project_id}/undo')
    def undo(project_id: str, request: RevisionRequest, time: int = 0, slice: int | None = None):
        project = store.get(project_id)
        with project.lock:
            project.check_revision(request.revision)
            project.check_slice(time, project.voxels.shape[1] // 2 if slice is None else slice)
            project.undo()
            return project.view(time, slice)

    @application.post('/api/projects/{project_id}/paired-annotation')
    def load_paired_annotation(project_id: str, request: RevisionRequest):
        project = store.get(project_id)
        entry, labels = read_annotation(ROOT, project.image, project.name)
        with project.lock:
            editable(project, request.revision)
            if any(project.locks.values()):
                raise ValueError('Unlock all structures before loading the matched annotation.')
            planes = project.unlocked_planes()
            times = sorted({t for t, _ in planes})
            keys = [(t, z, label) for t, z in planes for label in ORGANS]
            snapshot = project.snapshot_planes(keys, 'Load matched refinement annotation')
            for t, z, label in keys:
                project.masks[label][t, z] = labels[t, z] == label
                project.contexts[(t, z, label)] = {'points': [], 'prior': None, 'mask_prior': True,
                                                 'iteration': 1, 'mode': None}
            project.commit(snapshot, 'import_paired_annotation', filename=entry['filename'], sha256=entry['sha256'],
                           changed_times=times, skipped_locked_frames=sorted(project.locked_frames),
                           skipped_locked_slices=[list(pair) for pair in sorted(project.locked_slices)])
            return project.view()

    @application.post('/api/projects/{project_id}/redo')
    def redo(project_id: str, request: RevisionRequest, time: int = 0, slice: int | None = None):
        project = store.get(project_id)
        with project.lock:
            project.check_revision(request.revision)
            project.check_slice(time, project.voxels.shape[1] // 2 if slice is None else slice)
            project.undo(redo=True)
            return project.view(time, slice)

    @application.post('/api/projects/{project_id}/clear-slice')
    def clear_slice(project_id: str, request: SliceRequest):
        project = store.get(project_id)
        with project.lock:
            editable(project, request.revision, request.organ, request.time, request.slice)
            project.check_slice(request.time, request.slice, request.organ)
            key = (request.time, request.slice, request.organ)
            snapshot = project.snapshot(key, 'Clear structure on this slice')
            project.masks[request.organ][request.time, request.slice].fill(0)
            project.contexts[key] = {'points': [], 'prior': None, 'iteration': 0, 'mode': None}
            project.commit(snapshot, 'clear_slice', time_frame=request.time, slice=request.slice, organ=request.organ)
            return project.view(request.time, request.slice)

    def clear_all_masks(project_id, request, all_times):
        project = store.get(project_id)
        with project.lock:
            editable(project, request.revision)
            project.check_slice(request.time, request.slice)
            if any(project.locks.values()):
                raise ValueError('Unlock all structures before clearing all masks.')
            if not all_times:
                project.check_plane_unlocked(request.time, request.slice)
            planes = project.unlocked_planes(slices=[request.slice]) if all_times else [(request.time, request.slice)]
            times = [t for t, _ in planes]
            scope = (None if all_times else request.time, request.slice, tuple(ORGANS))
            description = 'Clear all masks on this layer across time' if all_times else 'Clear all masks on this slice'
            snapshot = (project.snapshot_planes([(t, request.slice, label) for t in times for label in ORGANS], description)
                        if all_times and project.locked_slices else project.snapshot(scope, description))
            for t in times:
                for label in ORGANS:
                    project.masks[label][t, request.slice].fill(0)
                    project.contexts[(t, request.slice, label)] = {
                        'points': [], 'prior': None, 'iteration': 0, 'mode': None}
            details = {'slice': request.slice, 'time_frames': len(times), 'changed_times': list(times),
                       'skipped_locked_frames': sorted(project.locked_frames)} if all_times else {
                'time_frame': request.time, 'slice': request.slice}
            if all_times:
                details['skipped_locked_slices'] = [list(pair) for pair in sorted(project.locked_slices)
                                                   if pair[1] == request.slice]
            project.commit(snapshot, 'clear_all_masks_layer' if all_times else 'clear_all_masks_slice', **details)
            return project.view(request.time, request.slice)

    @application.post('/api/projects/{project_id}/clear-all-masks-slice')
    def clear_all_masks_slice(project_id: str, request: SegmentAllRequest):
        return clear_all_masks(project_id, request, False)

    @application.post('/api/projects/{project_id}/clear-all-masks-layer')
    def clear_all_masks_layer(project_id: str, request: SegmentAllRequest):
        return clear_all_masks(project_id, request, True)

    @application.post('/api/projects/{project_id}/clear-points')
    def clear_points(project_id: str, request: SliceRequest):
        project = store.get(project_id)
        with project.lock:
            editable(project, request.revision, request.organ, request.time, request.slice)
            project.check_slice(request.time, request.slice, request.organ)
            key = (request.time, request.slice, request.organ)
            snapshot = project.snapshot(key, 'Clear prompts and continue from current mask')
            project.contexts[key] = manual_context(project.masks[request.organ][request.time, request.slice],
                                                   project.context(key)['mode'])
            project.commit(snapshot, 'clear_points', time_frame=request.time, slice=request.slice, organ=request.organ)
            return project.view(request.time, request.slice)

    @application.post('/api/projects/{project_id}/lock')
    def lock_structure(project_id: str, request: LockRequest):
        project = store.get(project_id)
        with project.lock:
            project.check_revision(request.revision)
            project.locks[request.organ] = request.locked
            project.changed('lock' if request.locked else 'unlock', organ=request.organ)
            return project.view()

    @application.post('/api/projects/{project_id}/frame-locks')
    def frame_locks(project_id: str, request: FrameLockRequest):
        project = store.get(project_id)
        project.set_frame_locks(request.revision, request.times, request.locked)
        with project.lock:
            return project.view()

    @application.post('/api/projects/{project_id}/slice-locks')
    def slice_locks(project_id: str, request: SliceLockRequest):
        project = store.get(project_id)
        project.set_slice_locks(request.revision, request.times, request.slices, request.locked)
        with project.lock:
            return project.view()

    @application.post('/api/projects/{project_id}/resolve-overlaps')
    def resolve_overlaps(project_id: str, request: ResolveRequest):
        project = store.get(project_id)
        with project.lock:
            editable(project, request.revision, time=request.time, slice=request.slice)
            project.check_slice(request.time, request.slice)
            labels = tuple(label for label in ORGANS if not project.locks[label])
            if not labels:
                raise ValueError('Unlock a structure before resolving overlaps.')
            snapshot = project.snapshot((request.time, request.slice, labels), 'Resolve overlapping structures on this slice')
            locked = [label for label in ORGANS if project.locks[label]]
            priority = locked + [request.priority] + list(ORGANS)
            occupied = np.zeros(project.voxels.shape[2:], dtype=bool)
            visited = set()
            for label in priority:
                if label in visited:
                    continue
                visited.add(label)
                volume_mask = project.masks[label][request.time, request.slice]
                old = volume_mask.copy()
                if not project.locks[label]:
                    volume_mask[occupied] = 0
                occupied |= volume_mask.astype(bool)
                if np.any(old != volume_mask):
                    key = (request.time, request.slice, label)
                    project.contexts[key] = manual_context(volume_mask, project.context(key)['mode'])
            project.commit(snapshot, 'resolve_overlaps', time_frame=request.time, slice=request.slice, priority=request.priority)
            return project.view(request.time, request.slice)

    @application.post('/api/projects/{project_id}/annotations')
    def import_annotation(project_id: str, file: UploadFile = File(...),
                          revision: int = Form(...), target: Literal['annotation', 'reference'] = Form('annotation'),
                          binary_organ: int = Form(0)):
        project = store.get(project_id)
        path, filename = upload_path(file)
        try:
            try:
                annotation = sitk.ReadImage(str(path))
            except RuntimeError as error:
                raise ValueError('The annotation could not be read as a NIfTI image.') from error
            require_same_geometry(project.image, annotation)
            labels = array4d(annotation)
            if not np.isin(labels, [0, 1, 2, 3]).all():
                raise ValueError('Use integer labels 0, 1 (RV), 2 (MYO), 3 (LV), or a binary mask.')
            if binary_organ:
                if binary_organ not in ORGANS or not np.isin(labels, [0, 1]).all():
                    raise ValueError('Choose a valid structure for a binary annotation.')
                labels = labels.astype(np.uint8) * binary_organ
            with project.lock:
                editable(project, revision)
                if target == 'reference':
                    project.reference = labels.astype(np.uint8)
                    project.changed('import_reference', filename=filename)
                else:
                    planes = project.unlocked_planes()
                    times = sorted({t for t, _ in planes})
                    organs = [label for label in ORGANS if not project.locks[label]
                              and (not binary_organ or label == binary_organ)]
                    if not organs:
                        raise ValueError('Unlock a structure before importing its annotation.')
                    keys = [(t, z, label) for t, z in planes for label in organs]
                    snapshot = project.snapshot_planes(keys, 'Import labelmap')
                    protected = np.zeros(project.voxels.shape, dtype=bool)
                    for label in ORGANS:
                        if project.locks[label]:
                            protected |= project.masks[label].astype(bool)
                    for label in organs:
                        for t, z in planes:
                            project.masks[label][t, z] = ((labels[t, z] == label) & ~protected[t, z]).astype(np.uint8)
                            key = (t, z, label)
                            project.contexts[key] = manual_context(project.masks[label][t, z])
                    project.commit(snapshot, 'import_annotation', filename=filename, binary_organ=binary_organ,
                                   changed_times=times, skipped_locked_frames=sorted(project.locked_frames),
                                   skipped_locked_slices=[list(pair) for pair in sorted(project.locked_slices)])
                return project.view()
        finally:
            path.unlink(missing_ok=True)

    @application.post('/api/projects/{project_id}/save')
    def save(project_id: str, request: RevisionRequest):
        project = store.get(project_id)
        with project.lock:
            project.check_revision(request.revision)
        project.save()
        with project.lock:
            return project.view()

    @application.get('/api/projects/{project_id}/export')
    def export(project_id: str, kind: str = 'bundle'):
        project = store.get(project_id)
        if kind == 'study':
            if project.pending_job:
                raise ValueError('Wait for inference before exporting a study backup.')
            path = create_backup(project)
            return FileResponse(path, media_type='application/zip', filename='Cardiac-Prism_study_backup.zip',
                                headers={'Content-Encoding': 'identity'},
                                background=BackgroundTask(path.unlink, missing_ok=True))
        with project.lock:
            data, media_type, filename = project.export(kind)
            return Response(data, media_type=media_type,
                            headers={'Content-Disposition': f'attachment; filename="Cardiac-Prism_{filename}"'})

    frontend = Path(os.environ.get('CARDIAC_PRISM_FRONTEND_DIR', ROOT / 'frontend/dist')).resolve()
    if frontend.exists():
        application.mount('/', StaticFiles(directory=frontend, html=True), name='workstation')
    return application


app = create_app()
