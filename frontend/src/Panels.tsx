import { useState } from 'react';
import {
  Eye, EyeOff, LockKeyhole, LockKeyholeOpen, ChevronDown, BrainCircuit,
  WandSparkles, Download, Upload, ScanLine, Layers3, Check, CircleHelp,
  CirclePlus, CircleMinus, Crosshair, Hand, ZoomIn, Contrast, Brush, Eraser,
} from 'lucide-react';
import type { Demo, GPUDevice, Health, LayerSettings, Project, Tool, StudyNotes as Notes } from './types';
import StudyNotes from './StudyNotes';
import { countFrameLocks, countLayerLocks, isSliceLocked, lockedSlices } from './lockScope';
import GPUSelector, { type GPUSelection } from './GPUSelector';

export const tools: { id: Tool; name: string; key: string; icon: typeof CirclePlus }[] = [
  { id: 'positive', name: 'Include', key: '1', icon: CirclePlus },
  { id: 'negative', name: 'Exclude', key: '2', icon: CircleMinus },
  { id: 'crosshair', name: 'Locate', key: '3', icon: Crosshair },
  { id: 'brush', name: 'Brush', key: '4', icon: Brush },
  { id: 'erase', name: 'Erase', key: '5', icon: Eraser },
  { id: 'pan', name: 'Pan', key: '6', icon: Hand },
  { id: 'zoom', name: 'Zoom', key: '7', icon: ZoomIn },
  { id: 'window', name: 'Window', key: '8', icon: Contrast },
];

type LeftProps = {
  project: Project; health: Health | null; selected: number; layers: LayerSettings; busy: boolean;
  onSelect: (id: number) => void; onLayers: (layers: LayerSettings) => void;
  onLock: (id: number, locked: boolean) => void;
  onDevice: (device: string) => void;
  beforeComplete: () => Promise<void>; registerNotesSave: (save: (() => Promise<void>) | null) => void;
  onNotesSaved: (id: string, notes: Notes) => void;
  onNotesDirty: (id: string, dirty: boolean) => void;
  managingNotes: boolean; registerNotesUpdate: (update: ((notes: Notes) => void) | null) => void;
};

export function StudyPanel({ project, health, selected, layers, busy, onSelect, onLayers, onLock, onDevice, beforeComplete, registerNotesSave, onNotesSaved, onNotesDirty, managingNotes, registerNotesUpdate }: LeftProps) {
  const [details, setDetails] = useState(false);
  const [pointControls, setPointControls] = useState(false);
  const inferenceStatus = !health ? 'Connecting…' : health.model.state === 'loading' ? 'Preparing inference…' :
    health.model.state === 'switching' ? 'Switching device…' : null;
  return <aside className="study-panel">
    <div className="panel-block study-identity">
      <div className="section-label">CURRENT STUDY <span className="tiny-tag">{project.provenance.dicom_import ? 'DICOM' : 'NIfTI'}</span></div>
      <div className="study-thumbnail"><ScanLine size={21} /><span>CARDIAC<br /><strong>MR IMAGING</strong></span></div>
      <h2 title={project.name}>{project.name.replace('.nii.gz', '').replace('.nii', '')}</h2>
      <div className="study-subtitle">{project.dimensions[2]} slices <span>·</span> {project.shape[0]} time {project.shape[0] === 1 ? 'frame' : 'frames'}</div>
      <button className="disclosure" onClick={() => setDetails(!details)} aria-expanded={details}>
        Image geometry <ChevronDown size={13} className={details ? 'rotated' : ''} />
      </button>
      {details && <dl className="metadata-grid">
        <dt>Dimensions</dt><dd>{project.dimensions.join(' × ')}</dd>
        <dt>Spacing</dt><dd>{project.spacing.map((value) => value.toFixed(2)).join(' × ')} mm</dd>
        <dt>Coordinates</dt><dd>LPS, physical space</dd>
        <dt>Source</dt><dd>{project.source_geometry.dimension}D scalar image</dd>
      </dl>}
      {details && project.provenance.note && <p className="muted-help" title={project.provenance.note}>{project.provenance.note}</p>}
      {details && project.input_profile && <div className="input-profile" data-testid="input-profile">
        <strong>Training-aligned normalization</strong>
        <span>{project.input_profile.input_state === 'trusted_prepared' ? 'Trusted prepared image' : 'Raw image'} · input range checked at every inference</span>
        {!!project.input_profile.warnings.length && <span className="input-profile-warning">{project.input_profile.warnings.join(' ')}</span>}
      </div>}
    </div>
    {health?.features?.study_notes && <StudyNotes key={project.id} id={project.id} busy={busy} managing={managingNotes} registerUpdate={registerNotesUpdate}
      beforeComplete={beforeComplete} registerSave={registerNotesSave} onSaved={onNotesSaved} onDirty={onNotesDirty} />}
    <div className="inference-controls">
      <label className="field-label" htmlFor="inference-device">Inference device</label>
      <select id="inference-device" value={health?.model.requested_device === 'auto' || health?.model.requested_device === 'cuda' ?
        (health.model.available_devices.some((device) => device.id === 'cuda:0') ? 'cuda:0' : 'cpu') : health?.model.requested_device || 'cpu'}
        disabled={busy || !health || health.model.state === 'switching'} onChange={(event) => onDevice(event.target.value)}>
        {(health?.model.available_devices || [{ id: 'cpu', name: 'CPU' }]).map((device) => <option value={device.id} key={device.id}>{device.name}</option>)}
      </select>
      {inferenceStatus && <p className="inference-status" role="status">{inferenceStatus}</p>}
      {health?.model.error && <p className="inference-error" role="alert">Inference unavailable: {health.model.error}</p>}
    </div>
    <div className="panel-block structures-block">
      <div className="section-label">STRUCTURES <span>3</span></div>
      <div className="muted-help">Select a structure to edit</div>
      <div className="structure-list">
        {project.structures.map((structure) => <div className={`structure-row ${selected === structure.id ? 'selected' : ''}`} key={structure.id}
          style={{ '--structure-color': structure.color } as React.CSSProperties}>
          <button className="structure-select" onClick={() => onSelect(structure.id)} aria-label={`Select ${structure.short}`} aria-pressed={selected === structure.id}>
            <span className="structure-dot" /><span><strong>{structure.short}</strong><small>{structure.name}</small></span>
          </button>
          <div className="structure-controls">
            <button title={layers.visible[structure.id] ? 'Hide structure' : 'Show structure'} aria-label={`Toggle ${structure.short} visibility`}
              onClick={() => onLayers({ ...layers, visible: { ...layers.visible, [structure.id]: !layers.visible[structure.id] } })}>
              {layers.visible[structure.id] ? <Eye size={14} /> : <EyeOff size={14} />}
            </button>
            <button title={structure.locked ? 'Unlock structure' : 'Protect structure'} aria-label={`${structure.locked ? 'Unlock' : 'Lock'} ${structure.short}`}
              disabled={busy} onClick={() => onLock(structure.id, !structure.locked)}>
              {structure.locked ? <LockKeyhole size={13} /> : <LockKeyholeOpen size={13} />}
            </button>
          </div>
          <div className="structure-stat"><span>{structure.volume_ml.toFixed(1)} <small>mL</small></span><span>{structure.annotated_slices}/{project.dimensions[2]} slices</span></div>
        </div>)}
      </div>
      <div className="reference-row">
        <span className="reference-line" /><div><strong>Reference contours</strong><small>{project.has_reference ? 'Ground-truth comparison layer' : 'Import a reference to compare'}</small></div>
        <button disabled={!project.has_reference} className={layers.showReference ? 'active' : ''} aria-label="Toggle reference contours"
          onClick={() => onLayers({ ...layers, showReference: !layers.showReference })}>{layers.showReference ? <Eye size={15} /> : <EyeOff size={15} />}</button>
      </div>
    </div>
    <div className="panel-block display-block">
      <div className="section-label">OVERLAY DISPLAY</div>
      <div className="range-caption"><span>Mask opacity</span><span>{Math.round(layers.opacity * 100)}%</span></div>
      <input type="range" min="0" max="0.85" step="0.05" value={layers.opacity} aria-label="Mask opacity"
        onChange={(event) => onLayers({ ...layers, opacity: Number(event.target.value) })} />
      <label className="checkbox-label"><input type="checkbox" checked={layers.outlinesOnly}
        onChange={(event) => onLayers({ ...layers, outlinesOnly: event.target.checked })} /><span>Contours only</span></label>
      <div className="range-caption"><label htmlFor="locator-color">Locator color</label>
        <input id="locator-color" type="color" value={layers.crosshairColor} aria-label="Locator color"
          onChange={(event) => onLayers({ ...layers, crosshairColor: event.target.value })} /></div>
      <div className="range-caption"><span>Locator opacity</span><span>{Math.round(layers.crosshairOpacity * 100)}%</span></div>
      <input type="range" min="0" max="1" step="0.05" value={layers.crosshairOpacity} aria-label="Locator opacity"
        onChange={(event) => onLayers({ ...layers, crosshairOpacity: Number(event.target.value) })} />
      <button className="disclosure" aria-label="Adjust click marker opacity" aria-expanded={pointControls} onClick={() => setPointControls(!pointControls)}>
        Click markers · {Math.round(layers.pointOpacity * 100)}% <ChevronDown size={13} className={pointControls ? 'rotated' : ''} />
      </button>
      {pointControls && <div className="point-controls">
        <div className="range-caption"><span>Point opacity</span><button className="text-button" aria-label={layers.pointOpacity ? 'Hide click markers' : 'Show click markers'}
          onClick={() => onLayers({ ...layers, pointOpacity: layers.pointOpacity ? 0 : .6 })}>{layers.pointOpacity ? 'Hide' : 'Show'}</button></div>
        <input type="range" min="0" max="1" step="0.05" value={layers.pointOpacity} aria-label="Point opacity"
          onChange={(event) => onLayers({ ...layers, pointOpacity: Number(event.target.value) })} />
        <p className="muted-help">Changes marker visibility; your prompts stay active.</p>
      </div>}
      <p className="muted-help">The 3D locator shares the slice cursor. X and Y follow the scan grid; Z follows the acquisition layers. Locator opacity also controls these axes.</p>
      <div className="section-label">3D MASK DISPLAY</div>
      <div className="range-caption"><span>3D mask opacity</span><span>{Math.round(layers.surfaceOpacity * 100)}%</span></div>
      <input type="range" min="0" max="1" step="0.05" value={layers.surfaceOpacity} aria-label="3D mask opacity"
        onChange={(event) => onLayers({ ...layers, surfaceOpacity: Number(event.target.value) })} />
      <label className="checkbox-label"><input type="checkbox" checked={layers.smoothSurfaces}
        onChange={(event) => onLayers({ ...layers, smoothSurfaces: event.target.checked })} /><span>Smooth 3D surfaces</span></label>
      <p className="muted-help">Use Render 3D while paused. Playback, changing frames or editing masks releases the surfaces. Smoothing affects display only.</p>
    </div>
  </aside>;
}

type RightProps = {
  project: Project; selected: number; tool: Tool; mode: 'organ_click' | 'click';
  brushRadius: number; windowRange: [number, number]; busy: boolean;
  onMode: (mode: 'organ_click' | 'click') => void; onBrushRadius: (radius: number) => void;
  onWindow: (range: [number, number]) => void; onAutomatic: () => void;
  onClear: (kind: 'clear-points' | 'clear-slice' | 'clear-all-masks-slice' | 'clear-all-masks-layer') => void;
  onImport: (file: File, target: 'annotation' | 'reference', binaryOrgan: number) => void;
  onExport: (kind: string) => void; onResolve: () => void;
  backupEnabled: boolean;
  onDemoPredictions: () => void;
  pairedAnnotation?: Demo;
  onPairedAnnotation: () => void;
  onSegmentAll: () => void;
  onSegmentAllLayers: () => void;
  gpus: GPUDevice[];
  gpuSelection: GPUSelection;
  onSegmentAllFrames: (gpuIds?: number[]) => void;
};

export function Inspector(props: RightProps) {
  const [exportKind, setExportKind] = useState('bundle');
  const [importTarget, setImportTarget] = useState<'annotation' | 'reference'>('annotation');
  const [binaryOrgan, setBinaryOrgan] = useState(0);
  const [importOpen, setImportOpen] = useState(false);
  const chooseGPUs = props.gpuSelection.enabled, selectedGPUs = props.gpuSelection.selected;
  const structure = props.project.structures[props.selected - 1];
  const sliceLocked = isSliceLocked(props.project, props.project.time, props.project.slice);
  const lockedCount = lockedSlices(props.project).length;
  const totalSlices = props.project.shape[0] * props.project.dimensions[2];
  const allSlicesLocked = lockedCount === totalSlices;
  const frameLockCount = countFrameLocks(props.project, props.project.time);
  const frameFullyLocked = frameLockCount === props.project.dimensions[2];
  const layerLockCount = countLayerLocks(props.project, props.project.slice);
  const layerFullyLocked = layerLockCount === props.project.shape[0];
  const width = props.windowRange[1] - props.windowRange[0];
  const level = (props.windowRange[1] + props.windowRange[0]) / 2;
  const sourceWidth = props.project.display_window[1] - props.project.display_window[0];
  return <aside className="inspector-panel">
    <div className="panel-block">
      <div className="section-label">EDITING <span className="active-structure" style={{ color: structure.color }}>{structure.short}</span></div>
      <div className="tool-description"><span className={`tool-symbol ${props.tool}`}>
        {(() => { const Icon = tools.find((tool) => tool.id === props.tool)!.icon; return <Icon size={22} />; })()}
      </span><div><strong>{tools.find((tool) => tool.id === props.tool)!.name}</strong><small>{sliceLocked ? 'This slice is locked' : structure.locked ? 'Structure is locked' : 'Acquisition plane'}</small></div></div>
      {sliceLocked && <div className="frame-protection-note" data-testid="frame-protection-note"><LockKeyhole size={13} /><span>Frame {props.project.time + 1}, layer {props.project.slice + 1} is protected for RV + MYO + LV. Unlock it in Edit History to modify masks or prompts. Browsing and export remain available.</span></div>}
      <div className="instruction-card">{props.tool === 'positive' ? <>Click <strong>inside</strong> the structure to include it.<br /><span>Shift + click adds an exclusion point.</span></> :
        props.tool === 'negative' ? <>Click an <strong>unwanted region</strong> to exclude it.</> :
        props.tool === 'brush' || props.tool === 'erase' ? <>Drag to {props.tool === 'brush' ? 'paint' : 'erase'} this structure on this slice.
          {props.tool === 'brush' && <> Your selected structure, <strong>{structure.short}</strong>, has highest painting priority and replaces other unlocked structures at painted pixels.</>} Your edits become the next AI refinement input.</> :
        props.tool === 'crosshair' ? <>Click any slice view to locate the same point across all three planes.</> :
        <>Drag to {props.tool === 'pan' ? 'pan the image' : props.tool === 'zoom' ? 'adjust zoom' : 'adjust window and level'}. Scroll to browse slices.</>}</div>
      {(props.tool === 'brush' || props.tool === 'erase') && <><div className="range-caption"><span>{props.tool === 'erase' ? 'Eraser size' : 'Brush size'}</span>
        <span>{props.brushRadius * 2 + 1} {props.brushRadius === 0 ? 'pixel' : 'px'}</span></div>
        <input type="range" min="0" max="30" value={props.brushRadius} aria-label={props.tool === 'erase' ? 'Eraser size' : 'Brush size'}
          onChange={(event) => props.onBrushRadius(Number(event.target.value))} />
        <p className="muted-help">{props.brushRadius === 0 ? 'One original image pixel. Drag for a line one pixel wide.' : 'Size is measured in original image pixels.'}</p></>}
      <div className="prompt-counter"><span><i className="positive-dot" /> {structure.points.filter((point) => point.positive).length} include</span>
        <span><i className="negative-dot" /> {structure.points.filter((point) => !point.positive).length} exclude</span></div>
      <div className="small-button-row">
        <button disabled={props.busy || sliceLocked || structure.locked || !structure.points.length} onClick={() => props.onClear('clear-points')}>Clear points</button>
        <button disabled={props.busy || sliceLocked || structure.locked || !structure.voxel_count} onClick={() => props.onClear('clear-slice')}>Clear this slice</button>
      </div>
      <button className="secondary-button full-width segment-all-button" data-testid="clear-all-masks-slice"
        disabled={props.busy || sliceLocked || props.project.structures.some(item => item.locked)} onClick={() => props.onClear('clear-all-masks-slice')}>
        Clear all masks this slice</button>
      <p className="muted-help">RV + MYO + LV, slice {props.project.slice + 1}{props.project.shape[0] > 1 ? ` of frame ${props.project.time + 1}` : ''}. Clears masks and prompts together; undo is available.</p>
      <button className="secondary-button full-width" data-testid="clear-all-masks-layer"
        disabled={props.busy || layerFullyLocked || props.project.structures.some(item => item.locked)} onClick={() => props.onClear('clear-all-masks-layer')}>
        Clear all masks this layer</button>
      <p className="muted-help">{props.project.shape[0] > 1 ? `RV + MYO + LV on slice ${props.project.slice + 1} across ${props.project.shape[0] - layerLockCount} unlocked time frames. ${layerLockCount} locked slices are skipped.` : 'For a 3D scan, this clears the same current slice.'} Other layers stay unchanged. One undo restores the operation.</p>
      {props.project.structures.some(item => item.locked) && <p className="muted-help">Unlock all structures to clear all masks.</p>}
    </div>
    {props.pairedAnnotation && <div className="panel-block">
      <div className="section-label">PAIRED ANNOTATION <Layers3 size={13} /></div>
      <p className="muted-help" title={props.pairedAnnotation.annotation_name}>Matched CineMA masks · RV 1 / MYO 2 / LV 3. New library studies start from these editable masks.</p>
      <button className="secondary-button full-width" data-testid="load-paired-annotation"
        disabled={props.busy || allSlicesLocked || props.project.structures.some(item => item.locked)} onClick={props.onPairedAnnotation}>Load matched annotation</button>
      <p className="muted-help">Replaces editable masks and prompts on unlocked slices only; {lockedCount} locked slices are preserved. Use this to restart refinement of an existing study; undo restores your previous masks.</p>
    </div>}
    <div className="panel-block ai-block">
      <div className="section-label">AI ASSISTANCE <WandSparkles size={13} /></div>
      <label className="field-label" htmlFor="inference-mode">Click mode</label>
      <select id="inference-mode" value={props.mode} disabled={props.busy} onChange={(event) => props.onMode(event.target.value as 'organ_click' | 'click')}>
        <option value="organ_click">Organ + clicks</option><option value="click">Clicks only</option>
      </select>
      <p className="muted-help">{props.mode === 'click' ? 'Use point prompts without an organ query.' : 'Use the selected organ with your point prompts.'}</p>
      {(structure.mode === 'click' || structure.mode === 'organ_click') && structure.mode !== props.mode &&
        <p className="muted-help">The next click starts a fresh prompt history in this mode.</p>}
      <button className="primary-button full-width" disabled={props.busy || sliceLocked || structure.locked} onClick={props.onAutomatic} data-testid="automatic-segmentation">
        <BrainCircuit size={16} /> Segment this slice
      </button>
      <div className="subtle-note"><CircleHelp size={12} /><span>Runs organ-only initialization for {structure.short}.</span></div>
      <button className="secondary-button full-width segment-all-button" disabled={props.busy || sliceLocked || props.project.structures.every((item) => item.locked)}
        onClick={props.onSegmentAll} data-testid="segment-all"><Layers3 size={16} /> Segment all</button>
      <p className="muted-help">RV + MYO + LV on this slice only. Highest probability wins overlaps; locked structures are preserved. Replaces masks and prompts on this slice.</p>
      <button className="secondary-button full-width segment-all-button" disabled={props.busy || frameFullyLocked || props.project.structures.every((item) => item.locked)}
        onClick={props.onSegmentAllLayers} data-testid="segment-all-layers"><Layers3 size={16} /> Segment all layers</button>
      <p className="muted-help">RV + MYO + LV on {props.project.dimensions[2] - frameLockCount} unlocked layers{props.project.shape[0] > 1 ? ` of time frame ${props.project.time + 1}` : ''}.
        {frameLockCount} locked layers are skipped. Each layer runs independently. Highest probability wins overlaps. Undo is available.</p>
      {props.project.shape[0] > 1 && <div className="all-frames-controls">
        <GPUSelector gpus={props.gpus} selection={props.gpuSelection} disabled={props.busy} scope="all time frames" />
        <button className="primary-button full-width" disabled={props.busy || allSlicesLocked || props.project.structures.every((item) => item.locked) || (chooseGPUs && !selectedGPUs.length)}
          onClick={() => props.onSegmentAllFrames(chooseGPUs ? selectedGPUs : undefined)} data-testid="segment-all-time-frames">
          <Layers3 size={16} /> Segment all time frames
        </button>
        <p className="muted-help" data-testid="all-frames-lock-summary">Will process {totalSlices - lockedCount} slices within {props.project.shape[0]} study frames, RV + MYO + LV; skip {lockedCount} locked slices. Each 2D plane runs independently. Results appear after the complete batch finishes. Replaces unlocked masks and prompts; highest probability wins. Cancel preserves current annotations. Undo restores the whole operation.</p>
      </div>}
      {props.project.provenance.geometry_source && <div className="demo-predictions">
        <button className="secondary-button full-width" disabled={props.busy || allSlicesLocked} onClick={props.onDemoPredictions}>Load evaluated 3D example</button>
        <p className="muted-help">Precomputed AI masks from one native GT-sampled click per slice. Replaces unlocked masks; locked slices are preserved. Undo is available.</p>
      </div>}
      {props.project.slice_overlap_voxels > 0 && <div className="overlap-notice"><strong>{props.project.slice_overlap_voxels.toLocaleString('en-US')} overlapping pixels on this slice</strong>
        <p>Use Segment all to recompute by probability, or give the selected structure priority.</p>
        <button disabled={props.busy || sliceLocked} onClick={props.onResolve}>Resolve · prioritize {structure.short}</button>
      </div>}
    </div>
    <div className="panel-block window-block">
      <div className="section-label">IMAGE DISPLAY <Contrast size={13} /></div>
      <div className="range-caption"><span>Window</span><span>{Math.round(width)}</span></div>
      <input type="range" min="5" max={Math.max(510, sourceWidth * 2)} value={width} aria-label="Window width"
        onChange={(event) => { const next = Number(event.target.value); props.onWindow([level - next / 2, level + next / 2]); }} />
      <div className="range-caption"><span>Level</span><span>{Math.round(level)}</span></div>
      <input type="range" min={props.project.display_window[0] - sourceWidth / 2} max={props.project.display_window[1] + sourceWidth / 2}
        value={level} aria-label="Window level" onChange={(event) => { const next = Number(event.target.value); props.onWindow([next - width / 2, next + width / 2]); }} />
      <button className="text-button" onClick={() => props.onWindow(props.project.display_window)}>Reset display</button>
    </div>
    <div className="panel-block export-block">
      <div className="section-label">ANNOTATIONS <Layers3 size={13} /></div>
      <button className="secondary-button full-width" disabled={props.busy} onClick={() => setImportOpen(!importOpen)}><Upload size={14} /> Import annotation</button>
      {importOpen && <div className="import-controls">
        <select aria-label="Import destination" value={importTarget} onChange={(event) => setImportTarget(event.target.value as 'annotation' | 'reference')}>
          <option value="annotation">Editable annotation</option><option value="reference">Reference comparison</option>
        </select>
        <select aria-label="Annotation format" value={binaryOrgan} onChange={(event) => setBinaryOrgan(Number(event.target.value))}>
          <option value="0">Labelmap: RV 1 / MYO 2 / LV 3</option>
          {props.project.structures.map((item) => <option key={item.id} value={item.id}>Binary mask → {item.short}</option>)}
        </select>
        <label className="file-button">Choose NIfTI file<input type="file" accept=".nii,.gz" disabled={props.busy || (importTarget === 'annotation' && allSlicesLocked)} aria-label="Import NIfTI annotation"
          onChange={(event) => { const file = event.target.files?.[0]; if (file) props.onImport(file, importTarget, binaryOrgan); event.target.value = ''; }} /></label>
        {importTarget === 'annotation' && lockedCount > 0 && <p className="muted-help">Import updates unlocked slices only. {lockedCount} locked slices keep their masks, clicks and priors.</p>}
      </div>}
      <label className="field-label" htmlFor="export-kind">Export format</label>
      <select id="export-kind" value={exportKind} onChange={(event) => setExportKind(event.target.value)}>
        {props.backupEnabled && <option value="study">Full study backup: image, masks, notes & prompts (.zip)</option>}
        <option value="bundle">Separate masks + edit record (.zip)</option><option value="labelmap">Combined labelmap (.nii.gz)</option>
        {props.project.structures.map((item) => <option key={item.id} value={String(item.id)}>{item.short} binary mask (.nii.gz)</option>)}
      </select>
      <button className="secondary-button full-width" disabled={props.busy} onClick={() => props.onExport(exportKind)} data-testid="export-button"><Download size={14} /> Export annotations</button>
      <div className="subtle-note"><Check size={12} /><span>Original geometry and time dimensions preserved.</span></div>
    </div>
  </aside>;
}
