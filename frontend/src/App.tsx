import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, ArrowLeft, ArrowRight, Check, ChevronDown, CircleHelp,
  FolderOpen, Keyboard, LoaderCircle, Save, Scan, Undo2, Redo2, Upload,
  X, AlertCircle, RotateCcw, Layers3, Clock3, Play, Pause, SkipBack, SkipForward, Repeat, Trash2, ArchiveRestore,
} from 'lucide-react';
import { api, APIError } from './api';
import { FrameCache, frameChanges } from './frameCache';
import { StudyFrameCache } from './studyFrameCache';
import { mergeMaskSlice } from './maskSlice';
import { countFrameLocks, isSliceLocked, lockedSlices } from './lockScope';
import { Inspector, StudyPanel, tools } from './Panels';
import ViewportGrid from './ViewportGrid';
import AnnotationTimeline from './AnnotationTimeline';
import TemporalPropagation from './TemporalPropagation';
import InputFormatGuide from './InputFormatGuide';
import JobProgress from './JobProgress';
import type { MedicalViewer } from './viewer';
import type { AnnotationProgress, Demo, GPUDevice, Health, Index3, Job, LayerSettings, Point, Project, SavedStudy, Tool, StudyNotes, PropagationRequest } from './types';

const initialLayers: LayerSettings = { visible: { 1: true, 2: true, 3: true }, opacity: .3,
  outlinesOnly: false, showReference: false, showContext: false, contextOpacity: .2, smoothSurfaces: false, surfaceOpacity: 1, pointOpacity: .6,
  crosshairColor: '#00e5ff', crosshairOpacity: .9, cineContext: false, cineSmoothSurfaces: false,
  showSlicePlane: false, slicePlaneOpacity: .45 };
type Data = { image: Float32Array; masks: Uint8Array; reference: Uint8Array | null };
type QueuedClick = { point: Point; id: string; time: number; slice: number; organ: number; mode: 'organ_click' | 'click' };
type LocalRefresh = { metadata: Project; previousRevision: number; time: number; slice: number };
type DeleteTarget = { kind: 'study'; id: string; name: string } | { kind: 'example'; name: string };

type Preferences = { selected: number; tool: Tool; mode: 'organ_click' | 'click'; time: number;
  position: Index3; layers: LayerSettings; windowRange: [number, number]; brushRadius: number };
function readPreferences(id: string): Partial<Preferences> {
  try { return JSON.parse(localStorage.getItem(`kprism-view-${id}`) || '{}') || {}; } catch { return {}; }
}

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [data, setData] = useState<Data | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [gpus, setGpus] = useState<GPUDevice[]>([]);
  const [chooseGPUs, setChooseGPUs] = useState(false);
  const [chosenGPUs, setChosenGPUs] = useState<number[] | null>(null);
  const selectedGPUs = (chosenGPUs ?? gpus.filter(gpu => gpu.available).slice(0, 1).map(gpu => gpu.id))
    .filter(id => gpus.some(gpu => gpu.id === id && gpu.available));
  const gpuSelection = { enabled: chooseGPUs, selected: selectedGPUs, onEnabled: setChooseGPUs, onSelected: setChosenGPUs };
  const [demos, setDemos] = useState<Demo[]>([]);
  const [studies, setStudies] = useState<SavedStudy[]>([]);
  const [annotationProgress, setAnnotationProgress] = useState<AnnotationProgress | null>(null);
  const [progressFailure, setProgressFailure] = useState('');
  const [time, setTime] = useState(0);
  const [position, setPosition] = useState<Index3>([0, 0, 0]);
  const [selected, setSelected] = useState(3);
  const [tool, setTool] = useState<Tool>('positive');
  const [mode, setMode] = useState<'organ_click' | 'click'>('organ_click');
  const [brushRadius, setBrushRadius] = useState(0);
  const [layers, setLayers] = useState<LayerSettings>(initialLayers);
  const [windowRange, setWindowRange] = useState<[number, number]>([0, 255]);
  const [loading, setLoading] = useState('Opening workspace');
  const [editing, setEditing] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [pollingWarning, setPollingWarning] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [lastTiming, setLastTiming] = useState<number | null>(null);
  const [notice, setNotice] = useState<{ type: 'error' | 'success'; text: string } | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [libraryTab, setLibraryTab] = useState<'examples' | 'saved' | 'completed' | 'removed'>('examples');
  const [libraryAction, setLibraryAction] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const folderInput = useRef<HTMLInputElement>(null);
  const dicomFolderInput = useRef<HTMLInputElement>(null);
  const [dicomUploadProgress, setDicomUploadProgress] = useState<number | null>(null);
  const notesSave = useRef<(() => Promise<void>) | null>(null);
  const notesUpdate = useRef<((notes: StudyNotes) => void) | null>(null);
  const [dirtyNoteId, setDirtyNoteId] = useState<string | null>(null);
  const notesDirty = useCallback((id: string, dirty: boolean) => { setDirtyNoteId(value => dirty ? id : value === id ? null : value); }, []);
  const registerNotesSave = useCallback((save: (() => Promise<void>) | null) => { notesSave.current = save; }, []);
  const registerNotesUpdate = useCallback((update: ((notes: StudyNotes) => void) | null) => { notesUpdate.current = update; }, []);
  const notesSaved = useCallback((id: string, notes: StudyNotes) => {
    setStudies(previous => previous.map(study => study.id === id ? { ...study, review_status: notes.review_status,
      note_preview: notes.text.slice(0, 160), notes_updated_at: notes.updated_at } : study));
  }, []);
  const [search, setSearch] = useState('');
  const [viewerReady, setViewerReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(10);
  const [loop, setLoop] = useState(true);
  const [buffering, setBuffering] = useState('');
  const [actualFps, setActualFps] = useState<number | null>(null);
  const frameCache = useRef<FrameCache | null>(null);
  const studyFrames = useRef(new StudyFrameCache());
  const awaitingViewer = useRef<string | null>(null);
  const playingRef = useRef(false);
  const playbackGeneration = useRef(0);
  const playbackOptions = useRef({ fps, loop }); playbackOptions.current = { fps, loop };
  const playbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pending, setPending] = useState<{ point: Point; id: string; time: number; slice: number; organ: number } | null>(null);
  const [queuedClicks, setQueuedClicks] = useState<QueuedClick[]>([]);
  const clickQueue = useRef<QueuedClick[]>([]);
  const clickRunning = useRef(false);
  const clickGeneration = useRef(0);
  const saving = useRef(false);
  const [saveEpoch, setSaveEpoch] = useState(0);
  const viewer = useRef<MedicalViewer | null>(null);
  const projectRef = useRef(project); projectRef.current = project;
  const dataRef = useRef(data); dataRef.current = data;
  const navigation = useRef({ time, position }); navigation.current = { time, position };
  const jobRef = useRef<string | null>(null);
  const loadingSequence = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const busy = !!loading || editing || !!job || playing || !!buffering || !!libraryAction;
  const current = useRef({ busy, selected, tool, mode }); current.current = { busy, selected, tool, mode };

  function error(value: unknown) {
    setNotice({ type: 'error', text: value instanceof Error ? value.message : String(value) });
  }

  function pausePlayback() {
    viewer.current?.setPlayback(false);
    playingRef.current = false; playbackGeneration.current++;
    if (playbackTimer.current !== null) clearTimeout(playbackTimer.current);
    playbackTimer.current = null; setPlaying(false); setBuffering('');
  }

  const loadStudy = useCallback(async (metadata: Project, frame = 0, preservePosition = false) => {
    clickGeneration.current++; clickQueue.current = []; setQueuedClicks([]);
    clickRunning.current = false; setPending(null); setEditing(false);
    const previousJob = jobRef.current;
    if (previousJob !== metadata.pending_job) {
      jobRef.current = null; setJob(null); setPollingWarning('');
      if (previousJob) void api.cancel(previousJob).catch(() => {});
    }
    pausePlayback(); setActualFps(null);
    const sequence = ++loadingSequence.current;
    setLoading('Loading image and masks');
    const preferences = preservePosition ? {} : readPreferences(metadata.id);
    if (Number.isInteger(preferences.time) && preferences.time! >= 0 && preferences.time! < metadata.shape[0]) frame = preferences.time!;
    const defaultPosition = [(metadata.dimensions[0] - 1) / 2, (metadata.dimensions[1] - 1) / 2, metadata.slice] as Index3;
    const rememberedPosition = preferences.position;
    const nextPosition = preservePosition ? navigation.current.position :
      Array.isArray(rememberedPosition) && rememberedPosition.length === 3 && rememberedPosition.every((value, axis) =>
        Number.isFinite(value) && value >= 0 && value <= metadata.dimensions[axis] - 1) ?
        [rememberedPosition[0], rememberedPosition[1], Math.round(rememberedPosition[2])] as Index3 : defaultPosition;
    try {
      const cached = studyFrames.current.get(metadata.id, frame);
      const sameRevision = cached?.project.revision === metadata.revision;
      const [image, maskFrame, reference, currentMetadata] = await Promise.all([
        cached ? Promise.resolve(cached.image) : api.volume(metadata.id, frame),
        cached && sameRevision ? Promise.resolve({ masks: cached.masks, revision: cached.project.revision }) : api.maskFrame(metadata.id, frame),
        metadata.has_reference ? (cached?.reference && sameRevision ? Promise.resolve(cached.reference) : api.reference(metadata.id, frame)) : Promise.resolve(null),
        api.project(metadata.id, frame, nextPosition[2]),
      ]);
      if (sequence !== loadingSequence.current) return;
      if (maskFrame.revision !== currentMetadata.revision) throw new Error('The study changed while opening. Please open it again.');
      const masks = maskFrame.masks;
      if (projectRef.current && dataRef.current) studyFrames.current.remember({ ...dataRef.current, project: projectRef.current });
      const switching = projectRef.current?.id !== currentMetadata.id;
      awaitingViewer.current = switching ? currentMetadata.id : null;
      setLoading(switching ? 'Preparing linked views' : '');
      if (switching) setViewerReady(false);
      frameCache.current = new FrameCache(metadata.id);
      frameCache.current.seed({ image, masks, reference, project: currentMetadata });
      setProject(currentMetadata); setData({ image, masks, reference }); setTime(frame); setPosition(nextPosition);
      setWindowRange(metadata.display_window); setLibraryOpen(false);
      if (!preservePosition) {
        const selected = [1, 2, 3].includes(preferences.selected || 0) ? preferences.selected! : 3;
        setSelected(selected);
        setTool(tools.some((tool) => tool.id === preferences.tool) ? preferences.tool! : 'positive');
        setBrushRadius(Number.isInteger(preferences.brushRadius) && preferences.brushRadius! >= 0 && preferences.brushRadius! <= 30 ? preferences.brushRadius! : 0);
        setMode(preferences.mode === 'click' || preferences.mode === 'organ_click' ? preferences.mode :
          currentMetadata.structures[selected - 1].mode === 'click' ? 'click' : 'organ_click');
        const candidate = preferences.layers ? { ...initialLayers, ...preferences.layers } : null;
        if (candidate && /^#[0-9a-f]{6}$/i.test(candidate.crosshairColor) && [candidate.opacity, candidate.contextOpacity, candidate.surfaceOpacity, candidate.pointOpacity, candidate.slicePlaneOpacity, candidate.crosshairOpacity].every((n) => Number.isFinite(n) && n >= 0 && n <= 1) &&
          [candidate.outlinesOnly, candidate.showReference, candidate.showContext, candidate.smoothSurfaces, candidate.showSlicePlane,
           candidate.cineContext, candidate.cineSmoothSurfaces, candidate.visible?.[1], candidate.visible?.[2], candidate.visible?.[3]].every((value) => typeof value === 'boolean')) setLayers({ ...candidate,
             showContext: false, showSlicePlane: false, cineContext: false, cineSmoothSurfaces: false });
        else setLayers(initialLayers);
        const range = preferences.windowRange;
        if (Array.isArray(range) && range.length === 2 && range.every(Number.isFinite) && range[1] > range[0]) setWindowRange(range);
        setLastTiming(null);
      }
      localStorage.setItem('kprism-last-study', metadata.id);
      const query = new URLSearchParams(location.search); query.set('study', metadata.id);
      window.history.replaceState(null, '', `?${query}`);
    } catch (value) {
      if (sequence === loadingSequence.current) { awaitingViewer.current = null; error(value); setLoading(''); }
    }
  }, []);

  useEffect(() => {
    if (!project || !viewerReady || loading || playing) return;
    try {
      localStorage.setItem(`kprism-view-${project.id}`, JSON.stringify({ selected, tool, mode, time, position, layers, windowRange, brushRadius }));
    } catch { /* Browsers with disabled storage can still edit server projects. */ }
  }, [project?.id, viewerReady, loading, playing, buffering, selected, tool, mode, time, position, layers, windowRange, brushRadius]);

  useEffect(() => {
    let active = true;
    void Promise.all([api.demos(), api.studies(), api.health()]).then(async ([examples, saved, status]) => {
      if (!active) return;
      setDemos(examples); setStudies(saved); setHealth(status);
      if (status.features?.study_lifecycle) void api.demos(true).then(setDemos);
      const queryParameters = new URLSearchParams(location.search);
      if (queryParameters.get('library') === '1') {
        setLoading(''); setLibraryOpen(true); return;
      }
      const query = queryParameters.get('study');
      const remembered = localStorage.getItem('kprism-last-study');
      const requested = query || (saved.some((study) => study.id === remembered) ? remembered : saved[0]?.id);
      try {
        const study = requested ? await api.project(requested) : null;
        if (!active) return;
        if (study) await loadStudy(study); else { setLoading(''); setLibraryOpen(true); }
      } catch (value) { error(value); setLoading(''); setLibraryOpen(true); }
    }).catch((value) => { if (active) { error(value); setLoading(''); } });
    const updateStatus = () => {
      void api.health().then((value) => { if (active) setHealth(value); }).catch(() => {});
      void api.gpus().then((value) => { if (active) setGpus(value); }).catch(() => {});
    };
    updateStatus();
    const interval = setInterval(updateStatus, 4000);
    return () => { active = false; clearInterval(interval); };
  }, [loadStudy]);

  useEffect(() => {
    if (!project) return;
    const id = project.id, frame = time, slice = position[2];
    let active = true;
    const timer = setTimeout(() => {
      void api.project(id, frame, slice).then((value) => {
        if (!active || projectRef.current?.id !== id || navigation.current.time !== frame || navigation.current.position[2] !== slice) return;
        if (current.current.busy) return;
        if (value.revision > projectRef.current.revision) {
          if (applyMetadataOnly(value)) return;
          pausePlayback(); void refresh(id).catch(error); return;
        }
        setProject((previous) => previous && value.revision >= previous.revision ? value : previous);
      }).catch((value) => { if (active) error(value); });
    }, 80);
    return () => { active = false; clearTimeout(timer); };
  }, [project?.id, project?.revision, time, position[2]]);

  useEffect(() => {
    if (!project || !health?.features?.annotation_progress) { setAnnotationProgress(null); return; }
    const id = project.id, revision = project.revision;
    let active = true;
    setProgressFailure('');
    setAnnotationProgress(previous => previous?.study_id === id ? previous : null);
    void api.progress(id).then(value => {
      if (!active || projectRef.current?.id !== id || value.revision !== revision || projectRef.current.revision !== revision) return;
      setAnnotationProgress(value);
    }).catch(value => { if (active) setProgressFailure(value instanceof Error ? value.message : String(value)); });
    return () => { active = false; };
  }, [project?.id, project?.revision, health?.features?.annotation_progress]);

  // Lock changes update annotation metadata only. Keep cine masks, compressed
  // frame archives and the current 3D surface intact.
  function applyMetadataOnly(metadata: Project) {
    const previous = projectRef.current;
    if (!previous || previous.id !== metadata.id || metadata.revision < previous.revision) return false;
    const changes = frameChanges(previous, metadata);
    if (!changes.frames || changes.frames.length || changes.referenceChanged) return false;
    frameCache.current?.invalidate(metadata.revision, []);
    projectRef.current = metadata; setProject(metadata);
    return true;
  }

  const refresh = useCallback(async (id?: string, changedReference = false, hint?: LocalRefresh): Promise<void> => {
    const currentProject = projectRef.current;
    if (!currentProject || (id && currentProject.id !== id)) return;
    const frame = navigation.current.time, slice = navigation.current.position[2];
    const scopedChanges = hint ? frameChanges(currentProject, hint.metadata) : null;
    const local = !changedReference && dataRef.current && hint?.metadata.id === currentProject.id &&
      hint.previousRevision === currentProject.revision && hint.metadata.revision === currentProject.revision + 1 &&
      hint.time === frame && hint.slice === slice && !scopedChanges?.referenceChanged &&
      scopedChanges?.frames?.length === 1 && scopedChanges.frames[0] === frame;
    const [metadata, maskFrame, reference] = await Promise.all([
      local ? Promise.resolve(hint!.metadata) : api.project(currentProject.id, frame, slice),
      local ? api.maskSlice(currentProject.id, frame, slice) : api.maskFrame(currentProject.id, frame),
      changedReference ? api.reference(currentProject.id, frame) : Promise.resolve(undefined),
    ]);
    if (projectRef.current?.id !== metadata.id || navigation.current.time !== frame) return;
    if (maskFrame.revision !== metadata.revision) {
      if (local) { await refresh(id); return; }
      throw new Error('The study changed while reloading. Pause and retry.');
    }
    const changes = frameChanges(currentProject, metadata);
    const nextReference = reference === undefined && changes.referenceChanged ?
      (metadata.has_reference ? await api.reference(currentProject.id, frame) : null) : reference;
    if (projectRef.current?.id !== metadata.id || navigation.current.time !== frame || projectRef.current.revision > metadata.revision) return;
    frameCache.current?.invalidate(metadata.revision, changes.frames, changedReference || changes.referenceChanged);
    viewer.current?.invalidateFrames(metadata.revision, changes.frames);
    const partial = 'slice' in maskFrame && maskFrame.slice !== null;
    if (partial && maskFrame.slice !== slice) throw new Error('The mask belongs to a different slice.');
    const masks = partial ? mergeMaskSlice(dataRef.current!.masks, maskFrame.masks, metadata.dimensions, slice) : maskFrame.masks;
    const updatedReference = nextReference === undefined ? dataRef.current?.reference ?? null : nextReference;
    if (dataRef.current) frameCache.current?.seed({ image: dataRef.current.image, masks, reference: updatedReference, project: metadata });
    projectRef.current = metadata;
    setProject((previous) => previous && previous.revision > metadata.revision ? previous : metadata);
    setData((previous) => previous ? { ...previous, masks, reference: nextReference === undefined ? previous.reference : nextReference } : previous);
  }, []);

  // Cancel scheduled playback on unmount. Every asynchronous frame request
  // also checks its generation before committing the shared cursor.
  useEffect(() => () => {
    playingRef.current = false; playbackGeneration.current++;
    jobRef.current = null;
    if (playbackTimer.current !== null) clearTimeout(playbackTimer.current);
  }, []);

  // A saved generation includes masks, prompts, priors, model provenance and
  // geometry. Save responses never replace a newer in-memory edit revision.
  useEffect(() => {
    if (!project?.dirty || busy || saving.current) return;
    const id = project.id, revision = project.revision;
    const timer = setTimeout(() => {
      saving.current = true;
      void api.edit(project, 'save').then((saved) => {
        setProject((value) => value?.id === id ? { ...value, saved_revision: saved.saved_revision,
          dirty: value.revision !== saved.saved_revision } : value);
      }).catch(error).finally(() => { saving.current = false; setSaveEpoch(value => value + 1); });
    }, 12000);
    return () => clearTimeout(timer);
  }, [project?.id, project?.revision, project?.saved_revision, project?.dirty, busy, saveEpoch]);

  useEffect(() => {
    if (!notice || notice.type === 'error') return;
    const timer = setTimeout(() => setNotice(null), 4500); return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (!job) return;
    const started = job.created_at ? Date.parse(job.created_at) : Date.now();
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - (Number.isFinite(started) ? started : Date.now())) / 1000)));
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, [job?.id, job?.created_at]);

  useEffect(() => {
    if (loading || !project?.pending_job || jobRef.current) return;
    const captured = project, taskId = project.pending_job;
    jobRef.current = taskId;
    setJob({ id: taskId, project_id: captured.id, time: captured.time, slice: captured.slice, organ: 0,
      state: 'running', seconds: 0, error: null });
    void (async () => {
      try {
        const status = await waitForJob(taskId, true);
        if (!status || projectRef.current?.id !== captured.id) return;
        if (status.state === 'failed') throw new Error(status.error ?? 'Inference failed.');
        if (status.state === 'stale') throw new Error('The study changed while inference ran. The outdated result was discarded.');
        await refresh(captured.id);
        if (status.state === 'completed') {
          setLastTiming(status.seconds);
          setNotice({ type: 'success', text: 'Segmentation completed. The full result is available; undo restores the operation.' });
        }
      } catch (value) { error(value); }
      finally {
        if (jobRef.current === taskId) { jobRef.current = null; setJob(null); setPollingWarning(''); }
      }
    })();
  }, [project?.id, project?.pending_job, loading]);

  async function openExample(name: string) {
    pausePlayback();
    setLoading('Opening cardiac MRI example'); setNotice(null);
    try {
      const available = await api.studies(); setStudies(available);
      const existing = available.find(study => study.name === name);
      await loadStudy(existing ? await api.project(existing.id) : await api.demo(name));
    } catch (value) { error(value); setLoading(''); }
  }

  async function uploadImage(file: File) {
    if (current.current.busy) return;
    if (file.name.toLowerCase().endsWith('.zip') && !health?.features?.study_lifecycle) {
      error('Full study backup import needs the updated backend. NIfTI scans can be opened now.'); return;
    }
    pausePlayback();
    setLoading(file.name.toLowerCase().endsWith('.zip') ? 'Restoring full study backup' : 'Reading NIfTI image'); setNotice(null);
    try { const study = await api.upload(file); setStudies(await api.studies()); await loadStudy(study); setFolderFiles([]); }
    catch (value) { error(value); setLoading(''); }
  }

  async function uploadDicomFolder(files: File[]) {
    if (!files.length || current.current.busy) return;
    if (!health?.features?.dicom_import) { error('DICOM folder import needs the updated backend. You can open NIfTI images now.'); return; }
    const folderName = files[0].webkitRelativePath.split('/')[0] || 'DICOM study';
    pausePlayback(); setNotice(null); setLoading('Uploading DICOM folder');
    try {
      const study = await api.importDicom(files, folderName, percent => {
        setDicomUploadProgress(percent);
        setLoading(percent === null ? 'Checking DICOM geometry and converting' : `Uploading DICOM folder · ${percent}%`);
      });
      setStudies(await api.studies()); setFolderFiles([]);
      const [width, height, layers] = study.dimensions;
      const warnings = study.provenance.dicom_import?.warnings || [];
      setNotice({ type: 'success', text: `DICOM imported: ${width} × ${height} · ${layers} layers · ${study.shape[0]} frame${study.shape[0] === 1 ? '' : 's'}. Saved in My studies.${warnings.length ? ' ' + warnings.join(' ') : ''}` });
      await loadStudy(study);
    } catch (value) { error(value); setLoading(''); }
    finally { setDicomUploadProgress(null); }
  }

  async function exampleVisibility(name: string, hidden: boolean) {
    setLoading(hidden ? 'Removing example from library' : 'Restoring example to library');
    try {
      await api.exampleVisibility(name, hidden); setDemos(await api.demos(true));
      setNotice({ type: 'success', text: hidden ? 'Example removed. Saved annotations are retained; restore it from Removed examples.' : 'Example restored.' });
    } catch (value) { error(value); } finally { setLoading(''); }
  }

  async function toggleStudyComplete(study: SavedStudy) {
    if (current.current.busy) return;
    const activeStudy = projectRef.current?.id === study.id;
    setLibraryAction('Saving completion status'); setNotice(null);
    try {
      if (activeStudy) await notesSave.current?.();
      const [metadata, notes] = await Promise.all([api.project(study.id), api.notes(study.id)]);
      const status = study.review_status === 'complete' ? 'in_progress' : 'complete';
      const saved = await api.completeStudy(metadata, notes, status);
      if (activeStudy && projectRef.current?.id === study.id) {
        notesUpdate.current?.(saved);
        const currentProject = projectRef.current;
        if (status === 'complete') {
          projectRef.current = { ...currentProject, saved_revision: metadata.revision, dirty: false };
          setProject(projectRef.current);
        }
      }
      setStudies(await api.studies());
      setNotice({ type: 'success', text: status === 'complete' ? 'Annotation completed. Masks and notes are saved.' : 'Study marked in progress.' });
    } catch (value) { error(value); } finally { setLibraryAction(''); }
  }

  async function confirmDelete() {
    const target = deleteTarget;
    if (!target || current.current.busy) return;
    setLibraryAction(target.kind === 'study' ? 'Deleting saved study' : 'Permanently removing example'); setNotice(null);
    try {
      if (target.kind === 'example') {
        await api.deleteExample(target.name); setDemos(await api.demos(true));
        setNotice({ type: 'success', text: 'Example permanently removed from the library. Saved studies and source files are retained.' });
      } else {
        if (projectRef.current?.id === target.id) await notesSave.current?.();
        const [metadata, notes] = await Promise.all([api.project(target.id), api.notes(target.id)]);
        await api.deleteStudy(metadata, notes);
        studyFrames.current.forget(target.id);
        if (projectRef.current?.id === target.id) {
          pausePlayback(); loadingSequence.current++; clickGeneration.current++;
          clickQueue.current = []; clickRunning.current = false; setQueuedClicks([]); setPending(null);
          awaitingViewer.current = null; frameCache.current = null; jobRef.current = null; setJob(null);
          projectRef.current = null; dataRef.current = null; setProject(null); setData(null); setViewerReady(false);
          setDirtyNoteId(null); notesSave.current = null; notesUpdate.current = null;
          const query = new URLSearchParams(location.search); query.delete('study'); query.set('library', '1');
          window.history.replaceState(null, '', `?${query}`);
        }
        try {
          localStorage.removeItem(`kprism-view-${target.id}`); localStorage.removeItem(`kprism-note-draft-${target.id}`);
          if (localStorage.getItem('kprism-last-study') === target.id) localStorage.removeItem('kprism-last-study');
        } catch { /* Optional browser preferences. */ }
        setStudies(await api.studies());
        setNotice({ type: 'success', text: 'Saved study deleted. Original source files and library examples are retained.' });
      }
      setDeleteTarget(null);
    } catch (value) { error(value); } finally { setLibraryAction(''); }
  }

  async function saveBeforeComplete() {
    const captured = projectRef.current; if (!captured || current.current.busy) throw new Error('Wait for the current operation before marking complete.');
    setEditing(true);
    try {
      const saved = await api.edit(captured, 'save');
      if (saved.dirty) throw new Error('The annotation changed while saving. Save the latest edits before marking complete.');
      projectRef.current = { ...captured, saved_revision: saved.saved_revision, dirty: false };
      setProject(projectRef.current); setStudies(await api.studies());
    } finally { setEditing(false); }
  }

  async function openSaved(id: string) {
    pausePlayback();
    setLoading('Restoring saved study'); setNotice(null);
    try { await loadStudy(await api.project(id)); } catch (value) { error(value); setLoading(''); }
  }

  async function edit(action: string, details: Record<string, unknown> = {}) {
    const captured = projectRef.current;
    if (!captured || current.current.busy) return;
    if (['brush', 'clear-points', 'clear-slice', 'clear-all-masks-slice', 'resolve-overlaps'].includes(action) &&
      isSliceLocked(captured, navigation.current.time, navigation.current.position[2])) {
      error(new Error('This slice is locked. Unlock it in Edit History before editing.')); return;
    }
    setEditing(true); setNotice(null);
    try {
      if (action === 'save') await notesSave.current?.();
      const scope = { time: navigation.current.time, slice: navigation.current.position[2] };
      const metadata = await api.edit(captured, action, details, scope.time, scope.slice);
      if (!applyMetadataOnly(metadata)) await refresh(captured.id, false, ['brush', 'clear-slice', 'clear-all-masks-slice', 'clear-all-masks-layer'].includes(action) ?
        { metadata, previousRevision: captured.revision, ...scope } : undefined);
      if (action === 'save') { setStudies(await api.studies()); setNotice({ type: 'success', text: 'Study saved. Masks and click history can be restored.' }); }
    } catch (value) { error(value); await reconcileAfterError(captured); } finally { setEditing(false); }
  }

  async function setSliceLocks(times: number[], slices: number[], locked: boolean) {
    const captured = projectRef.current;
    if (!captured || current.current.busy || !times.length || !slices.length) return;
    setEditing(true); setNotice(null);
    try {
      await api.sliceLocks(captured, times, slices, locked);
      const scope = navigation.current;
      const metadata = await api.project(captured.id, scope.time, scope.position[2]);
      if (projectRef.current?.id !== captured.id) return;
      if (!applyMetadataOnly(metadata)) await refresh(captured.id);
      setNotice({ type: 'success', text: `${locked ? 'Locked' : 'Unlocked'} ${times.length * slices.length} slice${times.length * slices.length === 1 ? '' : 's'} · ${times.length} frame${times.length === 1 ? '' : 's'} × ${slices.length} layer${slices.length === 1 ? '' : 's'} · RV + MYO + LV.${locked ? ' Masks, clicks and priors are protected, including empty masks.' : ' These slices can be edited again.'}` });
    } catch (value) { error(value); await reconcileAfterError(captured); }
    finally { setEditing(false); }
  }

  async function reconcileAfterError(captured: Project) {
    try {
      if (projectRef.current?.id !== captured.id) return;
      const scope = navigation.current;
      const metadata = await api.project(captured.id, scope.time, scope.position[2]);
      if (!applyMetadataOnly(metadata) && metadata.revision > (projectRef.current?.revision ?? -1)) await refresh(captured.id);
    } catch { /* Keep the original action error visible. */ }
  }

  async function waitForJob(taskId: string, batch: boolean): Promise<Job | null> {
    let failures = 0;
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, failures ? Math.min(3000, failures * 750) : batch ? 350 : 80));
      if (jobRef.current !== taskId) return null;
      let status: Job;
      try { status = await api.job(taskId); }
      catch (value) {
        if (jobRef.current !== taskId) return null;
        if (value instanceof APIError && value.status < 500 && ![408, 429].includes(value.status)) throw value;
        failures++;
        setPollingWarning('Connection interrupted. Retrying status; inference continues on the server.');
        continue;
      }
      if (jobRef.current !== taskId) return null;
      failures = 0; setPollingWarning(''); setJob(status);
      if (status.state !== 'queued' && status.state !== 'running') return status;
    }
  }

  async function infer(inferenceMode: string, point?: Point, gpuIds?: number[], scope?: QueuedClick, propagation?: PropagationRequest) {
    const captured = projectRef.current;
    const allStructures = ['all', 'all_layers', 'all_frames', 'propagation'].includes(inferenceMode);
    if (!captured || (current.current.busy && !scope) || (scope && captured.id !== scope.id) ||
      (allStructures ? captured.structures.every((item) => item.locked) :
      captured.structures[(scope?.organ ?? current.current.selected) - 1].locked)) return false;
    const frame = scope?.time ?? navigation.current.time, slice = scope?.slice ?? navigation.current.position[2],
      organ = scope?.organ ?? current.current.selected;
    if (!['all_frames', 'all_layers', 'propagation'].includes(inferenceMode) && isSliceLocked(captured, frame, slice)) {
      error(new Error('This slice is locked. Unlock it in Edit History before segmenting.')); return false;
    }
    if (inferenceMode === 'all_layers' && countFrameLocks(captured, frame) === captured.dimensions[2]) {
      error(new Error('All layers in this frame are locked. Unlock a slice before segmenting.')); return false;
    }
    if (inferenceMode === 'all_frames' && lockedSlices(captured).length === captured.shape[0] * captured.dimensions[2]) {
      error(new Error('All slices are locked. Unlock a slice before segmenting.')); return false;
    }
    setEditing(true); setNotice(null);
    if (point) setPending({ point, id: captured.id, time: frame, slice, organ });
    let taskId: string | null = null;
    const generation = clickGeneration.current;
    let applied = false;
    try {
      if (propagation && (propagation.revision !== captured.revision || propagation.slice !== slice))
        throw new Error('The study or layer changed. Review the updated propagation preview before running.');
      const submitted = propagation ? await api.propagate(captured.id, propagation) :
        inferenceMode === 'all_frames' ? await api.inferAllFrames(captured, gpuIds) :
        inferenceMode === 'all_layers' ? await api.inferAllLayers(captured, frame) :
        inferenceMode === 'all' ? await api.inferAll(captured, frame, slice) :
        await api.infer(captured, frame, slice, organ, inferenceMode, point);
      taskId = submitted.id;
      if (scope && generation !== clickGeneration.current) { void api.cancel(taskId).catch(() => {}); return false; }
      jobRef.current = taskId; setJob(submitted); setPollingWarning(''); setEditing(false);
      const status = await waitForJob(taskId, allStructures);
      if (status) {
        if (status.state === 'failed') throw new Error(status.error ?? 'Inference failed.');
        if (status.state === 'stale') throw new Error('The study changed while inference ran. The outdated result was discarded.');
        if (status.state === 'completed') {
          setLastTiming(status.seconds); await refresh(captured.id);
          applied = true;
          if (point && status.point_foreground !== undefined) {
            const matched = status.point_foreground === point.positive;
            setNotice({ type: 'success', text: `${captured.structures[organ - 1].short} click processed on frame ${frame + 1}, slice ${slice + 1}. ${status.changed_voxels ?? 0} pixels changed.${matched ? '' : ' The prediction at this point has not followed the prompt yet; add another point or use the brush.'}` });
          }
          if (inferenceMode === 'propagation') setNotice({ type: 'success', text: `Propagated layer ${slice + 1} to ${status.total_frames} time frame${status.total_frames === 1 ? '' : 's'}. Reference frames, locked slices and other layers were preserved. Review the masks in cine; one undo restores this operation.` });
          else if (inferenceMode === 'all_frames') setNotice({ type: 'success', text: `Segmented ${status.total_frames} time frames (${status.total_slices} planes). Skipped ${status.skipped_locked_slices?.length ?? lockedSlices(captured).length} locked slices; locked structures were preserved. Undo is available.` });
          else if (inferenceMode === 'all_layers') setNotice({ type: 'success', text: `Segmented ${status.total_slices} unlocked layers in time frame ${frame + 1}; skipped ${status.skipped_locked_slices?.length ?? countFrameLocks(captured, frame)} locked layers. Locked structures were preserved; undo is available.` });
          else if (status.protected_voxels) setNotice({ type: 'success', text: `Updated ${allStructures ? 'structures on this slice' : captured.structures[organ - 1].short}; locked structures were protected.` });
        }
      }
      setHealth(await api.health());
      return applied;
    } catch (value) { error(value); await refresh(captured.id).catch(() => {}); return false; }
    finally {
      if (!taskId || jobRef.current === taskId) { setJob(null); jobRef.current = null; setPending(null); setEditing(false); setPollingWarning(''); }
    }
  }

  function queuePoint(point: Point) {
    const captured = projectRef.current;
    if (!captured || playingRef.current || loading || buffering || (current.current.busy && !clickRunning.current)) {
      error(new Error('Pause playback and wait for the current operation before adding a click.')); return;
    }
    const entry: QueuedClick = { point, id: captured.id, time: navigation.current.time,
      slice: navigation.current.position[2], organ: current.current.selected, mode: current.current.mode };
    if (isSliceLocked(captured, entry.time, entry.slice)) { error(new Error('This slice is locked. Unlock it in Edit History before adding a click.')); return; }
    if (captured.structures[entry.organ - 1].locked) { error(new Error('Unlock the selected structure before adding a click.')); return; }
    clickQueue.current.push(entry); setQueuedClicks([...clickQueue.current]);
    if (clickRunning.current) return;
    clickRunning.current = true;
    const generation = clickGeneration.current;
    void (async () => {
      try {
        while (generation === clickGeneration.current && clickQueue.current.length) {
          const next = clickQueue.current[0];
          const applied = await infer(next.mode, next.point, undefined, next);
          if (generation !== clickGeneration.current) break;
          if (!applied) {
            clickQueue.current = []; setQueuedClicks([]); break;
          }
          if (generation !== clickGeneration.current) break;
          clickQueue.current.shift(); setQueuedClicks([...clickQueue.current]);
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        }
      } finally { if (generation === clickGeneration.current) clickRunning.current = false; }
    })();
  }

  async function changeDevice(device: string) {
    if (current.current.busy) return;
    setEditing(true); setNotice(null);
    try { await api.device(device); setHealth(await api.health()); }
    catch (value) { error(value); setHealth(await api.health().catch(() => health)); }
    finally { setEditing(false); }
  }

  async function cancelInference() {
    clickGeneration.current++; clickQueue.current = []; setQueuedClicks([]);
    clickRunning.current = false;
    const id = jobRef.current; if (!id) return;
    try {
      await api.cancel(id); jobRef.current = null; setJob(null); setPending(null); setEditing(false); setPollingWarning('');
      await refresh(); setNotice({ type: 'success', text: 'Inference result cancelled. You can continue editing.' });
    } catch (value) { error(value); }
  }

  async function importAnnotation(file: File, target: 'annotation' | 'reference', binaryOrgan: number) {
    const captured = projectRef.current; if (!captured || current.current.busy) return;
    setEditing(true); setNotice(null);
    try {
      await api.annotation(captured, file, target, binaryOrgan); await refresh(captured.id, target === 'reference');
      if (target === 'reference') setLayers((value) => ({ ...value, showReference: true }));
      setNotice({ type: 'success', text: target === 'reference' ? 'Reference loaded and geometry verified.' : 'Annotation loaded and geometry verified.' });
    } catch (value) { error(value); } finally { setEditing(false); }
  }

  async function exportAnnotations(kind: string) {
    if (!project || busy) return;
    setEditing(true);
    try {
      await notesSave.current?.(); const filename = await api.download(project, kind);
      if (kind === 'study') {
        const saved = await api.project(project.id, navigation.current.time, navigation.current.position[2]);
        setProject(value => value?.id === saved.id ? { ...value, saved_revision: saved.saved_revision, dirty: value.revision !== saved.saved_revision } : value);
        setStudies(await api.studies());
      }
      setNotice({ type: 'success', text: `Exported ${filename}` });
    }
    catch (value) { error(value); } finally { setEditing(false); }
  }

  async function displayFrame(frame: number, generation: number) {
    const captured = projectRef.current, cache = frameCache.current, medical = viewer.current;
    if (!captured || !cache || !medical?.ready || frame < 0 || frame >= captured.shape[0]) return false;
    const next = await cache.get(captured, frame, navigation.current.position[2]);
    if (generation !== playbackGeneration.current || projectRef.current?.id !== captured.id) return false;
    cache.activate(frame);
    navigation.current = { ...navigation.current, time: frame };
    const latestProject = projectRef.current!;
    const metadata = { ...next.project, saved_revision: latestProject.saved_revision, dirty: latestProject.dirty,
      undo_count: latestProject.undo_count, redo_count: latestProject.redo_count, events: latestProject.events,
      locked_frames: latestProject.locked_frames, locked_slices: latestProject.locked_slices };
    setProject(metadata); setData(next); setTime(frame);
    // Wait for React's layout update and the render before scheduling a new frame.
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    return generation === playbackGeneration.current;
  }

  async function changeTime(frame: number) {
    if (!projectRef.current || loading || editing || job || !viewerReady) return;
    pausePlayback();
    if (frame === navigation.current.time) return;
    const generation = playbackGeneration.current;
    setBuffering(`Opening frame ${frame + 1}`); setActualFps(null);
    try { await displayFrame(frame, generation); }
    catch (value) { if (generation === playbackGeneration.current) error(value); }
    finally { if (generation === playbackGeneration.current) setBuffering(''); }
  }

  async function runPlayback(generation: number) {
    const captured = projectRef.current, cache = frameCache.current;
    if (!captured || !cache || !viewer.current) return;
    try {
      // Start with the next frame; reuse bounded caches on subsequent loops.
      // Do not rebuild every frame before resuming after a one-layer edit.
      if (generation !== playbackGeneration.current) return;
      setBuffering('');
      let sampleStart = performance.now(), sampleFrames = 0;
      const tick = async () => {
        if (generation !== playbackGeneration.current || !playingRef.current) return;
        const p = projectRef.current; if (!p || p.id !== captured.id) { pausePlayback(); return; }
        let next = navigation.current.time + 1;
        if (next >= p.shape[0]) {
          if (!playbackOptions.current.loop) { pausePlayback(); return; }
          next = 0;
        }
        const started = performance.now();
        try {
          if (!cache.hasFrame(p, next)) setBuffering(`Preparing frame ${next + 1}`);
          if (!await displayFrame(next, generation)) return;
          setBuffering(''); sampleFrames++;
          const ahead = (next + 1) % p.shape[0];
          void cache.get(projectRef.current!, ahead, navigation.current.position[2]).catch(() => {});
          const now = performance.now();
          if (now - sampleStart >= 1000) { setActualFps(sampleFrames * 1000 / (now - sampleStart)); sampleStart = now; sampleFrames = 0; }
          playbackTimer.current = setTimeout(() => { void tick(); }, Math.max(0, 1000 / playbackOptions.current.fps - (now - started)));
        } catch (value) {
          if (generation === playbackGeneration.current) { pausePlayback(); error(value); }
        }
      };
      await tick();
    } catch (value) { if (generation === playbackGeneration.current) { pausePlayback(); error(value); } }
  }

  function togglePlayback() {
    if (playingRef.current) { pausePlayback(); return; }
    const p = projectRef.current;
    if (!p || p.shape[0] < 2 || current.current.busy || !viewer.current?.ready) return;
    viewer.current.setPlayback(true);
    playingRef.current = true; setPlaying(true); setActualFps(null);
    const generation = ++playbackGeneration.current;
    // Restart at the first frame when a non-looping cine ended.
    if (!playbackOptions.current.loop && navigation.current.time === p.shape[0] - 1) {
      void displayFrame(0, generation).then((valid) => { if (valid) void runPlayback(generation); }).catch((value) => { pausePlayback(); error(value); });
    } else void runPlayback(generation);
  }

  function setLocation(next: Index3) {
    if (!project) return;
    setPosition(next.map((value, axis) => Math.max(0, Math.min(project.dimensions[axis] - 1, axis === 2 ? Math.round(value) : value))) as Index3);
  }

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (deleteTarget) {
        if (event.key === 'Escape') { event.preventDefault(); if (!libraryAction) setDeleteTarget(null); }
        if (event.key === 'Tab') {
          const buttons = [...document.querySelectorAll<HTMLButtonElement>('.delete-dialog button:not(:disabled)')];
          const first = buttons[0], last = buttons[buttons.length - 1];
          if (first && event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
          else if (last && !event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
        return;
      }
      if (libraryOpen || helpOpen) {
        if (event.key === 'Escape') { setLibraryOpen(false); setHelpOpen(false); }
        return;
      }
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) return;
      if (event.key === 'Escape') { pausePlayback(); setLibraryOpen(false); setHelpOpen(false); return; }
      if (event.code === 'Space') { event.preventDefault(); togglePlayback(); return; }
      if ((event.ctrlKey || event.metaKey) && ['s', 'z', 'y'].includes(event.key.toLowerCase())) {
        event.preventDefault();
        if (event.key.toLowerCase() === 's') void edit('save');
        else void edit(event.key.toLowerCase() === 'y' || event.shiftKey ? 'redo' : 'undo');
        return;
      }
      const choice = tools.find((candidate) => candidate.key === event.key);
      if (choice) setTool(choice.id);
      else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); const value = navigation.current;
        const p = projectRef.current;
        if (p) setPosition([value.position[0], value.position[1], Math.max(0, Math.min(p.dimensions[2] - 1, value.position[2] + (event.key === 'ArrowDown' ? 1 : -1)))]);
      } else if (event.key.toLowerCase() === 'r') viewer.current?.fit();
    };
    document.addEventListener('keydown', handler); return () => document.removeEventListener('keydown', handler);
  });

  const selectedStructure = project?.structures[selected - 1];
  const unknownDicomTiming = !!project?.provenance.dicom_import && project.shape[0] > 1 && !project.provenance.dicom_import.timing_known;
  const pendingPoint = pending && project?.id === pending.id && pending.time === time && pending.slice === position[2] && pending.organ === selected ? pending.point : null;
  const pendingPoints = queuedClicks.filter(click => click.id === project?.id && click.time === time && click.slice === position[2] && click.organ === selected).map(click => click.point);
  const sliceLocked = project ? isSliceLocked(project, time, position[2]) : false;
  const acceptsPoints = !sliceLocked && !loading && !buffering && !playing && (!busy || clickRunning.current);
  const completedStudies = studies.filter(study => study.review_status === 'complete');
  const filteredDemos = demos.filter(demo => !!demo.hidden === (libraryTab === 'removed') && [demo.name, demo.patient, demo.phase].join(' ').toLowerCase().includes(search.toLowerCase()));
  const filteredStudies = (libraryTab === 'completed' ? completedStudies : studies).filter(study => [study.name, study.note_preview].join(' ').toLowerCase().includes(search.toLowerCase()));

  return <div className="application">
    <header className="app-header">
      <div className="brand" aria-label="Cardiac Prism work station"><img src="/prism.svg" alt="" /><strong>Cardiac Prism work station</strong></div>
      <div className="header-divider" />
      <button className="study-switcher" disabled={!!loading || editing || !!job || !!libraryAction} onClick={() => { pausePlayback(); setLibraryOpen(true); void api.studies().then(setStudies); }}>
        <FolderOpen size={16} /><span>{project ? project.name.replace('.nii.gz', '').replace('.nii', '') : 'Open study'}</span><ChevronDown size={13} />
      </button>
      <div className="header-spacer" />
      {studies.find(study => study.id === project?.id)?.review_status === 'complete' && <span className="study-card-tag completed" data-testid="study-completed-badge"><Check size={12} /> Annotation completed</span>}
      <span className={`save-status ${project?.dirty || dirtyNoteId === project?.id ? 'unsaved' : ''}`}>
        {project?.dirty || dirtyNoteId === project?.id ? <Clock3 size={13} /> : <Check size={13} />}{project?.dirty || dirtyNoteId === project?.id ? 'Unsaved edits' : 'All changes saved'}
      </span>
      <button className="secondary-button header-save" disabled={!project || busy} onClick={() => void edit('save')}><Save size={14} /> Save study</button>
      <button className="icon-button" aria-label="Keyboard shortcuts" title="Keyboard shortcuts" onClick={() => setHelpOpen(true)}><Keyboard size={18} /></button>
      <span className="workspace-badge"><Activity size={12} /> RESEARCH WORKSPACE</span>
    </header>
    <div className="app-body">
      {project && data ? <>
        <StudyPanel project={project} health={health} selected={selected} layers={layers} busy={busy}
          beforeComplete={saveBeforeComplete} registerNotesSave={registerNotesSave} onNotesSaved={notesSaved} onNotesDirty={notesDirty}
          managingNotes={!!libraryAction} registerNotesUpdate={registerNotesUpdate}
          onSelect={(id) => { setSelected(id); if (tool === 'negative') setTool('positive'); }} onLayers={setLayers}
          onLock={(organ, locked) => void edit('lock', { organ, locked })} onDevice={(device) => void changeDevice(device)} />
        <main className="imaging-workspace">
          <div className="workspace-topline"><div><span className="section-label">SEGMENTATION WORKSPACE</span><span className="workspace-subtitle">Review anatomy. Refine with a click.</span></div>
            <span className="layout-label"><Layers3 size={13} /> 4-view layout</span></div>
          <div className="toolbar" role="toolbar" aria-label="Annotation tools">
            <div className="tool-group">{tools.map((item, index) => <button key={item.id} className={`tool-button ${tool === item.id ? 'active' : ''} ${index === 3 || index === 5 ? 'tool-separator' : ''}`}
              aria-label={`${item.name} (${item.key})`} aria-pressed={tool === item.id} title={`${item.name} · ${item.key}`}
              onClick={() => setTool(item.id)}><item.icon size={17} /><span>{item.name}</span></button>)}</div>
            <div className="history-tools"><button title="Undo · Ctrl+Z" aria-label="Undo edit" disabled={busy || !project.undo_count} onClick={() => void edit('undo')}><Undo2 size={17} /></button>
              <button title="Redo · Ctrl+Shift+Z" aria-label="Redo edit" disabled={busy || !project.redo_count} onClick={() => void edit('redo')}><Redo2 size={17} /></button>
              <button title="Fit all views · R" aria-label="Fit all views" onClick={() => viewer.current?.fit()}><RotateCcw size={15} /></button></div>
          </div>
          <div className="viewer-region">
            <ViewportGrid key={project.id} project={project} time={time} playing={playing} image={data.image} masks={data.masks} reference={data.reference}
              position={position} setPosition={setLocation} tool={tool} structure={selected} layers={layers} onLayers={setLayers} windowRange={windowRange}
              busy={busy} acceptsPoints={acceptsPoints} pendingPoints={pendingPoints.length ? pendingPoints : pendingPoint ? [pendingPoint] : []} brushRadius={brushRadius}
              onPoint={queuePoint}
              onStroke={(points, erase) => void edit('brush', { time, slice: position[2], organ: selected, points, erase, radius: brushRadius })}
              onReady={(medical) => {
                viewer.current = medical; setViewerReady(!!medical);
                if (medical && awaitingViewer.current === medical.project.id) { awaitingViewer.current = null; setLoading(''); }
              }} onError={(message) => { awaitingViewer.current = null; setLoading(''); error(new Error(message)); }} />
            {loading && <div className="loading-scrim" role="status" aria-live="polite" data-testid="study-loading">
              <LoaderCircle className="spin" size={25} /><strong>{loading}</strong>
              <span>Opening your study. Editing resumes when the linked views are ready.</span></div>}
            {job && <div className="inference-banner"><LoaderCircle size={16} className="spin" />
              {queuedClicks.length > 1 && <span data-testid="queued-click-count">{queuedClicks.length - 1} clicks queued</span>}
              <JobProgress job={job} elapsed={elapsed} pollingWarning={pollingWarning} />
              <button onClick={() => void cancelInference()}>Cancel result <X size={12} /></button></div>}
          </div>
          <div className="slice-navigator">
            <span className="navigator-label">ACQUISITION</span>
            <button aria-label="Previous slice" disabled={position[2] === 0} onClick={() => setLocation([position[0], position[1], position[2] - 1])}><ArrowLeft size={15} /></button>
            <input type="range" min="0" max={project.dimensions[2] - 1} step="1" value={position[2]} aria-label="Acquisition slice"
              onChange={(event) => setLocation([position[0], position[1], Number(event.target.value)])} />
            <button aria-label="Next slice" disabled={position[2] === project.dimensions[2] - 1} onClick={() => setLocation([position[0], position[1], position[2] + 1])}><ArrowRight size={15} /></button>
            <span className="slice-position"><strong>{position[2] + 1}</strong> / {project.dimensions[2]}</span>
          </div>
          {project.shape[0] > 1 && <div className={`cine-player ${playing ? 'playing' : ''}`} data-testid="cine-player">
            <span className="navigator-label">CINE</span>
            <button className="cine-play" aria-label={playing ? 'Pause cine' : 'Play cine'} title="Play / pause · Space"
              disabled={!viewerReady || !!loading || editing || !!job || (!playing && !!buffering)} onClick={togglePlayback}>
              {playing ? <Pause size={15} /> : <Play size={15} />}<span>{playing ? 'Pause' : 'Play'}</span></button>
            <button aria-label="Previous time frame" disabled={!!loading || editing || !!job || time === 0}
              onClick={() => void changeTime(time - 1)}><SkipBack size={14} /></button>
            <input id="time-frame" type="range" min="0" max={project.shape[0] - 1} step="1" value={time} aria-label="Time frame"
              disabled={!!loading || editing || !!job} onChange={(event) => void changeTime(Number(event.target.value))} />
            <button aria-label="Next time frame" disabled={!!loading || editing || !!job || time === project.shape[0] - 1}
              onClick={() => void changeTime(time + 1)}><SkipForward size={14} /></button>
            <span className="cine-position" data-testid="cine-frame">{time + 1} / {project.shape[0]}</span>
            <button aria-label="Loop cine" aria-pressed={loop} className={loop ? 'active' : ''} title="Loop playback"
              onClick={() => setLoop((value) => !value)}><Repeat size={15} /></button>
            <label className="cine-speed"><select aria-label="Playback speed" value={fps} onChange={(event) => setFps(Number(event.target.value))}>
              {[1, 5, 10, 15, 20, 25, 30].map((speed) => <option key={speed} value={speed}>{speed} fps</option>)}
            </select></label>
            <span className="cine-status" role="status">{buffering ? <><LoaderCircle size={12} className="spin" />{buffering}</> :
              playing ? `Playing${actualFps === null ? '' : ` · ${actualFps.toFixed(1)} fps`} · editing paused` : 'Paused · edit current frame'}</span>
          </div>}
          {unknownDicomTiming && <p className="dicom-timing-note" data-testid="dicom-timing-note">Frame interval unavailable · playback speed is adjustable</p>}
          {health?.features?.annotation_progress && <AnnotationTimeline key={project.id}
            studyId={project.id} revision={project.revision} progress={annotationProgress?.study_id === project.id ? annotationProgress : null} failure={progressFailure}
            time={time} slice={position[2]} frames={project.shape[0]} depth={project.dimensions[2]}
            timeSpacing={unknownDicomTiming ? 0 : project.source_geometry.spacing[3] ?? 0} playing={playing} disabled={!!loading || editing || !!job || !!libraryAction || !!buffering}
            lockedSlices={lockedSlices(project)} locksEnabled={!!health.features.slice_locks} activityEnabled={!!health.features.activity_history} lockDisabled={busy || !viewerReady}
            onSliceLocks={(times, slices, locked) => void setSliceLocks(times, slices, locked)} onSelectionMode={pausePlayback}
            onTime={frame => void changeTime(frame)} onSlice={slice => { pausePlayback(); setLocation([position[0], position[1], slice]); }} />}
          {health?.features?.temporal_propagation && project.shape[0] > 1 && <TemporalPropagation key={`${project.id}:${position[2]}`}
            project={project} time={time} slice={position[2]} progress={annotationProgress} disabled={busy || !viewerReady}
            playing={playing} modelReady={health.model.state === 'ready'} completed={studies.find(study => study.id === project.id)?.review_status === 'complete'}
            gpus={gpus} gpuSelection={gpuSelection} onRun={request => void infer('propagation', undefined, undefined, undefined, request)}
            onReopen={() => { const study = studies.find(item => item.id === project.id); if (study) void toggleStudyComplete(study); }} />}
        </main>
        <Inspector project={project} selected={selected} tool={tool} mode={mode} brushRadius={brushRadius} windowRange={windowRange} busy={busy}
          onDemoPredictions={() => void edit('demo-predictions')}
          pairedAnnotation={demos.find(example => example.name === project.name && example.has_annotation)}
          onPairedAnnotation={() => void edit('paired-annotation')}
          onMode={setMode} onBrushRadius={setBrushRadius} onWindow={setWindowRange} onAutomatic={() => void infer('organ')}
          onSegmentAll={() => void infer('all')}
          onSegmentAllLayers={() => void infer('all_layers')}
          gpus={gpus} gpuSelection={gpuSelection} onSegmentAllFrames={(ids) => void infer('all_frames', undefined, ids)}
          onClear={(action) => void edit(action, { time, slice: position[2], organ: selected })}
          onImport={(file, target, binaryOrgan) => void importAnnotation(file, target, binaryOrgan)} onExport={(kind) => void exportAnnotations(kind)} backupEnabled={!!health?.features?.study_lifecycle}
          onResolve={() => void edit('resolve-overlaps', { time, slice: position[2], priority: selected })} />
      </> : <main className="empty-workspace"><div className="empty-prism"><img src="/prism.svg" alt="" /></div>
        <span className="eyebrow">K-PRISM IMAGING WORKSTATION</span><h1>From images to anatomy.</h1><p>A focused workspace for interactive cardiac segmentation.</p>
        {loading ? <div className="empty-loading"><LoaderCircle className="spin" size={18} />{loading}</div> :
          <button className="primary-button" onClick={() => setLibraryOpen(true)}><FolderOpen size={17} /> Open a study</button>}
      </main>}
    </div>
    <footer className="app-statusbar">
      <span><i className={`status-dot ${viewerReady ? 'ready' : ''}`} />{viewerReady ? 'Viewer ready' : loading ? 'Loading study' : 'Preparing viewer'}</span>
      {project && <><span className="status-divider" /><span>{project.spacing.map((value) => value.toFixed(2)).join(' × ')} mm</span>
        <span className="status-divider" /><span>I {position[0].toFixed(1)} · J {position[1].toFixed(1)} · K {position[2]}</span></>}
      <span className="header-spacer" />
      {lastTiming !== null && <span>Last inference {lastTiming.toFixed(1)}s</span>}
      <span className="status-divider" /><span>{selectedStructure?.short ?? 'Cardiac MRI'} · {mode === 'click' ? 'Clicks only' : 'Organ + clicks'}</span>
      <span className="status-divider" /><span>Local workspace</span>
    </footer>
    {notice && <div className={`toast ${notice.type}`} role={notice.type === 'error' ? 'alert' : 'status'}>
      {notice.type === 'error' ? <AlertCircle size={18} /> : <Check size={18} />}<span>{notice.text}</span><button aria-label="Dismiss notification" onClick={() => setNotice(null)}><X size={15} /></button>
    </div>}
    <input ref={fileInput} className="hidden-input" type="file" accept={health?.features?.study_lifecycle ? '.nii,.gz,.zip' : '.nii,.gz'} aria-label="Upload NIfTI study" onChange={(event) => {
      const file = event.target.files?.[0]; if (file) void uploadImage(file); event.target.value = '';
    }} />
    <input ref={folderInput} className="hidden-input" type="file" multiple {...{ webkitdirectory: '', directory: '' }} aria-label="Browse NIfTI folder"
      onChange={event => {
        const files = [...(event.target.files || [])].filter(file => /\.nii(\.gz)?$/i.test(file.name));
        setFolderFiles(files); event.target.value = '';
        if (!files.length) setNotice({ type: 'error', text: health?.features?.dicom_import ? 'This folder contains no .nii / .nii.gz scans. For DICOM, choose Open DICOM folder.' : 'This folder contains no .nii / .nii.gz scans. DICOM import needs the updated backend.' });
      }} />
    {health?.features?.dicom_import && <input ref={dicomFolderInput} className="hidden-input" type="file" multiple {...{ webkitdirectory: '', directory: '' }} aria-label="Open DICOM folder"
      onChange={event => { const files = [...(event.target.files || [])]; event.target.value = ''; if (files.length) void uploadDicomFolder(files); }} />}
    {libraryOpen && <div className="modal-backdrop" onClick={() => project && setLibraryOpen(false)}><section className="study-modal" role="dialog" aria-modal="true" aria-label="Study library" onClick={(event) => event.stopPropagation()}>
      <div className="modal-heading"><div><span className="section-label">STUDY LIBRARY</span><h2>Choose your next study</h2></div><button className="icon-button" aria-label="Close study library" onClick={() => setLibraryOpen(false)}><X size={20} /></button></div>
      {(loading || libraryAction) && <div className="library-loading" role="status" aria-live="polite"><LoaderCircle className="spin" size={17} /><span>{loading || libraryAction}</span></div>}
      {dicomUploadProgress !== null && <progress className="dicom-upload-progress" value={dicomUploadProgress} max="100" aria-label="DICOM upload progress" />}
      <div className="upload-dropzone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void uploadImage(file); }}>
        <Upload size={23} /><div><strong>Open your NIfTI volume</strong><p>.nii / .nii.gz · 3D or 4D{health?.features?.study_lifecycle && ' · full study backup .zip'}</p></div>
        <button className="secondary-button" disabled={busy} onClick={() => fileInput.current?.click()}>Browse files</button>
        <button className="secondary-button" disabled={busy} onClick={() => folderInput.current?.click()}>Browse folder</button>
      </div>
      {health?.features?.dicom_import && <div className="dicom-folder-entry" data-testid="dicom-folder-entry">
        <FolderOpen size={21} aria-hidden="true" /><div><strong>Open a DICOM folder</strong><p>One classic MR sequence · 3D or 4D · image size, layers and frames detected automatically</p></div>
        <button className="secondary-button" disabled={busy} onClick={() => dicomFolderInput.current?.click()}>Open DICOM folder</button>
      </div>}
      <InputFormatGuide backupEnabled={!!health?.features?.study_lifecycle} dicomEnabled={!!health?.features?.dicom_import} />
      {folderFiles.length > 0 && <div className="folder-files"><p>Select the source image to open. Import its matching mask in the annotation panel.</p>
        {folderFiles.map((file, index) => <button key={index} disabled={busy} onClick={() => void uploadImage(file)}>{file.webkitRelativePath || file.name}<ArrowRight size={14} /></button>)}</div>}
      <div className="library-tabs"><button className={libraryTab === 'examples' ? 'active' : ''} onClick={() => setLibraryTab('examples')}>Cardiac examples <span>{demos.filter(demo => !demo.hidden).length}</span></button>
        <button className={libraryTab === 'saved' ? 'active' : ''} onClick={() => { setLibraryTab('saved'); void api.studies().then(setStudies); }}>My studies <span>{studies.length}</span></button>
        <button className={libraryTab === 'completed' ? 'active' : ''} onClick={() => { setLibraryTab('completed'); void api.studies().then(setStudies); }}>Completed <span>{completedStudies.length}</span></button>
        {health?.features?.study_lifecycle && <button className={libraryTab === 'removed' ? 'active' : ''} onClick={() => setLibraryTab('removed')}>Removed examples <span>{demos.filter(demo => demo.hidden).length}</span></button>}
        <input type="search" placeholder="Search studies…" aria-label="Search studies" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
      <div className="study-cards">{['examples', 'removed'].includes(libraryTab) ? filteredDemos.map((demo) => <div className="study-card-with-action" key={demo.name} data-example-name={demo.name}><button className="study-card" disabled={busy || demo.hidden} onClick={() => void openExample(demo.name)}>
        <span className="study-card-icon"><Scan size={22} /></span><div><strong>{demo.patient.replace('patient', 'Patient ')}</strong><p>{demo.phase} · {demo.time_frames ? `${demo.time_frames} frames · ${demo.slices} slices` : 'Cardiac MRI'}</p><small>{demo.name}</small></div>
        <span className="study-card-tag">{demo.has_annotation ? 'Refinement mask' : demo.has_reference ? 'Reference available' : demo.time_frames ? '4D NIfTI' : 'NIfTI'}</span>
        {studies.some(study => study.name === demo.name && study.review_status === 'complete') && <span className="study-card-tag completed">Completed</span>}<ArrowRight size={16} />
      </button><div className="library-card-actions">{health?.features?.study_lifecycle && <button className="library-card-action" disabled={busy} aria-label={`${demo.hidden ? 'Restore' : 'Remove'} example ${demo.name}`} title={demo.hidden ? 'Restore to Cardiac examples' : 'Remove from Cardiac examples; keep saved annotations'} onClick={() => void exampleVisibility(demo.name, !demo.hidden)}>{demo.hidden ? <ArchiveRestore size={15} /> : <Trash2 size={15} />}<span>{demo.hidden ? 'Restore' : 'Remove'}</span></button>}
        {demo.hidden && health?.features?.study_management && <button className="library-card-action danger" disabled={busy} aria-label={`Permanently remove example ${demo.name}`} onClick={() => setDeleteTarget({ kind: 'example', name: demo.name })}><Trash2 size={15} /><span>Permanently remove</span></button>}
      </div></div>) : filteredStudies.map((study) => <div className="study-card-with-action" key={study.id} data-study-id={study.id}><button className="study-card" disabled={busy} onClick={() => void openSaved(study.id)}>
        <span className="study-card-icon"><FolderOpen size={22} /></span><div><strong>{study.name.replace('.nii.gz', '')}</strong><p>Last edited {new Date(study.modified_at).toLocaleString('en-US')}</p>
          {study.note_preview && <p className="study-note-preview">{study.note_preview}</p>}</div>
        <span className={`study-card-tag ${study.review_status === 'complete' ? 'completed' : ''}`}>{study.review_status === 'complete' ? 'Completed' : study.dirty ? 'Open session' : 'Saved'}</span><ArrowRight size={16} />
      </button>{health?.features?.study_management && <div className="library-card-actions">
        <button className={`library-card-action ${study.review_status === 'complete' ? 'complete' : ''}`} disabled={busy} aria-label={`${study.review_status === 'complete' ? 'Mark in progress' : 'Mark complete'} ${study.name}`} onClick={() => void toggleStudyComplete(study)}>{study.review_status === 'complete' ? <RotateCcw size={15} /> : <Check size={15} />}<span>{study.review_status === 'complete' ? 'Reopen' : 'Complete'}</span></button>
        <button className="library-card-action danger" disabled={busy} aria-label={`Delete study ${study.name}`} onClick={() => setDeleteTarget({ kind: 'study', id: study.id, name: study.name })}><Trash2 size={15} /><span>Delete</span></button>
      </div>}</div>)}{(libraryTab === 'completed' || libraryTab === 'saved' ? filteredStudies.length : filteredDemos.length) === 0 && <p className="library-empty">{libraryTab === 'completed' ? 'Mark an annotation complete to find it here.' : libraryTab === 'saved' ? 'No saved studies.' : libraryTab === 'removed' ? 'No removed examples.' : 'No matching examples.'}</p>}</div>
      <div className="modal-footer"><CircleHelp size={13} /><span>Matched annotations are editable refinement baselines. Reference contours remain separate.</span></div>
    </section></div>}
    {deleteTarget && <div className="modal-backdrop delete-backdrop"><section className="delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-dialog-title" aria-describedby="delete-dialog-description">
      <h2 id="delete-dialog-title">{deleteTarget.kind === 'study' ? 'Delete saved study?' : 'Permanently remove example?'}</h2>
      <p className="delete-study-name">{deleteTarget.name}</p>
      <p id="delete-dialog-description">{deleteTarget.kind === 'study' ? 'This deletes this saved study, including its image copy, masks, clicks, notes and completion status. It cannot be restored from the library. Export a full study backup first if you need to keep it. Original source files and library examples remain.' : 'This example will disappear from both Cardiac examples and Removed examples. It cannot be restored from this menu. Saved studies, annotations and original source files remain.'}</p>
      <div className="delete-dialog-actions"><button className="secondary-button" autoFocus disabled={!!libraryAction} onClick={() => setDeleteTarget(null)}>Cancel</button>
        <button className="danger-button" disabled={busy} onClick={() => void confirmDelete()}>{libraryAction ? <LoaderCircle size={14} className="spin" /> : <Trash2 size={14} />}{deleteTarget.kind === 'study' ? 'Delete study' : 'Permanently remove'}</button></div>
    </section></div>}
    {helpOpen && <div className="modal-backdrop" onClick={() => setHelpOpen(false)}><section className="help-modal" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" onClick={(event) => event.stopPropagation()}>
      <div className="modal-heading"><h2>Work at your pace</h2><button className="icon-button" onClick={() => setHelpOpen(false)} aria-label="Close shortcuts"><X size={20} /></button></div>
      <div className="shortcut-list">{tools.map((item) => <div key={item.id}><span>{item.name}</span><kbd>{item.key}</kbd></div>)}
        <div><span>Exclusion point while including</span><kbd>Shift + click</kbd></div><div><span>Undo / redo</span><kbd>Ctrl + Z / Ctrl + Shift + Z</kbd></div>
        <div><span>Save study</span><kbd>Ctrl + S</kbd></div><div><span>Previous / next slice</span><kbd>↑ / ↓</kbd></div><div><span>Fit all views</span><kbd>R</kbd></div><div><span>Play / pause cine</span><kbd>Space</kbd></div>
        <div><span>Pan / window level</span><kbd>Middle / right drag</kbd></div></div>
    </section></div>}
  </div>;
}
