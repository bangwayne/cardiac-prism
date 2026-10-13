import { useEffect, useId, useMemo, useState } from 'react';
import { Check, ChevronRight, CopyCheck, LoaderCircle, LockKeyhole, Plus, X } from 'lucide-react';
import { api } from './api';
import { isSliceLocked } from './lockScope';
import GPUSelector, { type GPUSelection } from './GPUSelector';
import type { AnnotationProgress, GPUDevice, Project, PropagationPreview, PropagationRequest } from './types';

const organNames = ['RV', 'MYO', 'LV'];
const names = (bits: number) => organNames.filter((_, index) => bits & (1 << index)).join(', ');
type Props = { project: Project; time: number; slice: number; progress: AnnotationProgress | null;
  disabled: boolean; playing: boolean; modelReady: boolean; completed: boolean; gpus: GPUDevice[]; gpuSelection: GPUSelection;
  onRun: (request: PropagationRequest) => void; onReopen: () => void };

/** Keyed by study + layer in App: references cannot silently travel to another layer. */
export default function TemporalPropagation({ project, time, slice, progress, disabled, playing, modelReady,
  completed, gpus, gpuSelection, onRun, onReopen }: Props) {
  const [open, setOpen] = useState(false);
  const [references, setReferences] = useState<number[]>([]);
  const [organs, setOrgans] = useState([1, 2, 3]);
  const [overwrite, setOverwrite] = useState(false);
  const [previewState, setPreviewState] = useState<{ key: string; value?: PropagationPreview; error?: string } | null>(null);
  const [gpuOpen, setGpuOpen] = useState(false);
  const bodyId = useId();
  const request = useMemo<PropagationRequest>(() => ({ revision: project.revision, slice,
    reference_times: references, organs, target_times: null, overwrite_edited: overwrite,
    gpu_ids: gpuSelection.enabled ? gpuSelection.selected : null }),
  [project.revision, slice, references, organs, overwrite, gpuSelection.enabled, gpuSelection.selected.join(',')]);
  const requestKey = JSON.stringify(request);
  const valid = references.length > 0 && organs.length > 0 && (!gpuSelection.enabled || gpuSelection.selected.length > 0);
  const preview = previewState?.key === requestKey ? previewState.value : undefined;
  const failure = previewState?.key === requestKey ? previewState.error : undefined;
  const history = progress?.study_id === project.id && progress.revision === project.revision ? progress : null;
  const depth = project.dimensions[2];
  const activeMask = history?.mask_labels[time * depth + slice] ?? 0;
  const selectionDisabled = disabled || completed;
  useEffect(() => {
    if (!open || !valid || disabled || completed || !modelReady) return;
    let active = true;
    const timer = setTimeout(() => {
      void api.propagationPreview(project.id, request).then(value => {
        if (active) setPreviewState({ key: requestKey, value });
      }).catch(error => {
        if (active) setPreviewState({ key: requestKey, error: error instanceof Error ? error.message : String(error) });
      });
    }, 180);
    return () => { active = false; clearTimeout(timer); };
  }, [open, valid, disabled, completed, modelReady, project.id, requestKey]);

  function toggleReference(frame: number) {
    if (selectionDisabled) return;
    setReferences(previous => previous.includes(frame) ? previous.filter(value => value !== frame) :
      previous.length < 3 ? [...previous, frame].sort((a, b) => a - b) : previous);
  }
  const editedCount = new Set(preview?.skipped.filter(item => item.reason === 'edited').map(item => item.time)).size;
  const lockedNames = [...new Set(preview?.skipped.filter(item => item.reason === 'locked').map(item => organNames[item.organ - 1]))];
  const lockedFrameCount = new Set(preview?.skipped.filter(item => ['slice_locked', 'frame_locked'].includes(item.reason)).map(item => item.time)).size;
  return <section className={`temporal-propagation ${open ? 'expanded' : ''}`} data-testid="temporal-propagation"
    aria-label={`Time propagation for layer ${slice + 1}`}
    onKeyDown={event => { if (event.key === 'Escape') { setOpen(false); event.stopPropagation(); }
      if (event.code === 'Space' || event.key.startsWith('Arrow')) event.stopPropagation(); }}>
    <div className="propagation-heading">
      <button className="annotation-toggle" aria-expanded={open} aria-controls={bodyId} aria-label="Set up time propagation"
        onClick={() => setOpen(value => !value)}><ChevronRight size={14} className={open ? 'expanded' : ''} />
        <CopyCheck size={14} /><span className="annotation-title">PROPAGATE IN TIME</span></button>
      <span className="propagation-scope" data-testid="propagation-scope">Layer {slice + 1} only · {references.length ? `Reference frames ${references.map(value => value + 1).join(', ')}` : 'Select 1–3 reference frames'}</span>
      {!open && <button className="propagation-link" disabled={selectionDisabled || !modelReady} onClick={() => {
        setOpen(true); if (activeMask && !references.includes(time) && references.length < 3) toggleReference(time);
      }}>Use current frame <Plus size={12} /></button>}
    </div>
    {open && <div id={bodyId} className="propagation-body" role="region" aria-label="Time propagation settings">
      <div className="propagation-floating-heading"><strong>Propagate layer {slice + 1} in time</strong><button className="icon-button"
        aria-label="Close propagation settings" onClick={() => setOpen(false)}><X size={16} /></button></div>
      <p className="propagation-intro">Choose well-segmented frames below. Cardiac-Prism uses their editable masks to predict the other time frames on <strong>layer {slice + 1}</strong>. Locked slices can be references and are always protected as targets.</p>
      {completed && <div className="propagation-blocked" role="status">This study is marked complete. <button disabled={disabled} onClick={onReopen}>Reopen to propagate</button></div>}
      {playing && <div className="propagation-blocked" role="status">Pause cine to choose references or run propagation.</div>}
      {!modelReady && <div className="propagation-blocked" role="status">Waiting for the inference model to be ready.</div>}
      <div className="propagation-reference-tools">
        <span>REFERENCE FRAMES <strong data-testid="propagation-reference-count">{references.length} / 3</strong></span>
        <button disabled={selectionDisabled || !history || !activeMask || (!references.includes(time) && references.length === 3)}
          aria-label={references.includes(time) ? 'Remove current reference frame' : 'Use current frame as reference'}
          onClick={() => toggleReference(time)}>{references.includes(time) ? <X size={12} /> : <Plus size={12} />}Frame {time + 1}</button>
        {!!references.length && <button disabled={selectionDisabled} onClick={() => setReferences([])} aria-label="Clear reference frames">Clear selection</button>}
      </div>
      <div className="annotation-markers propagation-markers" role="group" aria-label={`Reference time frames on layer ${slice + 1}`}>
        {Array.from({ length: project.shape[0] }, (_, frame) => {
          const index = frame * depth + slice, mask = history?.mask_labels[index] ?? 0, edited = history?.edited_labels[index] ?? 0;
          const chosen = references.includes(frame);
          const frameLocked = isSliceLocked(project, frame, slice);
          return <button key={frame} className={`annotation-marker ${edited ? 'edited' : mask ? 'masked' : 'empty'} ${chosen ? 'reference-selected' : ''} ${frame === time ? 'current' : ''}`}
            data-testid={`propagation-reference-${frame}`} aria-label={`Use frame ${frame + 1} as reference`} aria-pressed={chosen}
            disabled={selectionDisabled || !history || (!chosen && (!mask || references.length === 3))} onClick={() => toggleReference(frame)}
            title={`Frame ${frame + 1}, layer ${slice + 1}\n${mask ? `Masks: ${names(mask)}` : 'No editable mask on this layer'}${edited ? `\nEdited: ${names(edited)}` : ''}${frameLocked ? '\nSlice locked: available as a reference; cannot be overwritten' : ''}\n${chosen ? 'Selected reference — click to remove' : 'Click to select as reference; navigate with the cine/history controls'}`}>
            {chosen && <Check size={11} />}{frame + 1}{frameLocked && <LockKeyhole size={10} />}</button>;
        })}
      </div>
      <div className="propagation-key"><span className="progress-key masked">Mask</span><span className="progress-key edited">Edited</span><span className="progress-key reference">Reference</span>
        <span>Frame selection does not move the view.</span></div>
      <div className="propagation-options">
        <div className="propagation-organs" role="group" aria-label="Structures to propagate">
          {project.structures.map(structure => <label key={structure.id} className={structure.locked ? 'locked' : ''} style={{ color: structure.color }}>
            <input type="checkbox" aria-label={`Propagate ${structure.short}`} checked={organs.includes(structure.id)} disabled={selectionDisabled || structure.locked}
              onChange={event => setOrgans(previous => event.target.checked ? [...previous, structure.id].sort() : previous.filter(value => value !== structure.id))} />
            {structure.short}{structure.locked && ' · locked'}</label>)}
        </div>
        <label className="checkbox-label"><input type="checkbox" checked={overwrite} disabled={selectionDisabled}
          onChange={event => setOverwrite(event.target.checked)} aria-label="Replace manual edits in target frames" /><span>Replace manual edits in target frames</span></label>
        <button className="propagation-link" disabled={disabled} aria-expanded={gpuOpen} onClick={() => setGpuOpen(value => !value)}>
          {gpuSelection.enabled ? `${gpuSelection.selected.length} GPU${gpuSelection.selected.length === 1 ? '' : 's'} selected` : 'Current inference device'}<ChevronRight size={12} className={gpuOpen ? 'expanded' : ''} /></button>
      </div>
      {overwrite && <p className="muted-help">Replace manual edits applies only to unlocked target slices. Slice locks always take priority.</p>}
      {gpuOpen && <div className="propagation-gpus"><GPUSelector gpus={gpus} selection={gpuSelection} disabled={selectionDisabled} scope="time propagation" />
        <p className="muted-help">GPU selection is shared with Segment all time frames.</p></div>}
      <div className="propagation-summary" data-testid="propagation-preview" aria-live="polite">
        {!references.length ? 'Select at least one frame containing an editable mask.' : !organs.length ? 'Select a structure to propagate.' :
          gpuSelection.enabled && !gpuSelection.selected.length ? 'Select at least one available GPU, or use the current device.' : failure ? <span className="note-error">{failure}</span> :
          preview ? <><strong>{preview.total_frames} target frame{preview.total_frames === 1 ? '' : 's'} · {preview.total_predictions} structure predictions</strong>
            {preview.total_frames > 0 && <span className="propagation-targets" title={preview.eligible.map(item => `Frame ${item.time + 1}: ${item.organs.map(organ => organNames[organ - 1]).join(', ')}`).join('\n')}>Frames {preview.eligible.map(item => item.time + 1).join(', ')}</span>}
            <span>Reference frames stay unchanged.{lockedFrameCount > 0 && ` Skipping ${lockedFrameCount} locked slice${lockedFrameCount === 1 ? '' : 's'}.`}{editedCount > 0 && ` Manual edits protected in ${editedCount} frame${editedCount === 1 ? '' : 's'}.`}{lockedNames.length > 0 && ` Locked structures: ${lockedNames.join(', ')}.`}</span>
            {!preview.total_frames && <span>No eligible targets. Choose another reference, unlock a target slice or structure, or review the manual-edit setting.</span>}
            {preview.warnings.map((warning, index) => <span className="propagation-warning" key={index}>{warning}</span>)}</> :
          completed ? 'Reopen this study to preview the targets.' : disabled ? 'Preview resumes when editing or playback finishes.' :
          <span><LoaderCircle size={12} className="spin" /> Checking target frames…</span>}
      </div>
      <div className="propagation-run-row"><span>Other layers stay unchanged. One undo restores this operation.</span>
        <button className="primary-button" data-testid="propagate-time" disabled={selectionDisabled || !modelReady || !valid || !preview?.total_predictions}
          onClick={() => onRun(request)}><CopyCheck size={14} />Propagate to {preview?.total_frames ?? 0} frame{preview?.total_frames === 1 ? '' : 's'}</button></div>
    </div>}
  </section>;
}
