import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { ChevronRight, LockKeyhole, LockKeyholeOpen, MousePointer2 } from 'lucide-react';
import ActivityHistory from './ActivityHistory';
import type { AnnotationProgress } from './types';

const organs = (bits: number) => ['RV', 'MYO', 'LV'].filter((_, index) => bits & (1 << index)).join(', ');
const dateFormat = new Intl.DateTimeFormat('en-US', { dateStyle: 'short', timeStyle: 'medium' });
const stamp = (value?: string | null) => value ? dateFormat.format(new Date(value)) : 'No edit timestamp';
const state = (mask: number, edited: number) => edited ? 'edited' : mask ? 'masked' : 'empty';
function ranges(values: number[]) {
  const result: string[] = [];
  for (let i = 0; i < values.length; i++) {
    const first = values[i]; let last = first;
    while (values[i + 1] === last + 1) last = values[++i];
    result.push(first === last ? String(first + 1) : `${first + 1}–${last + 1}`);
  }
  return result.join(', ');
}
function choose(values: number[], value: number, anchor: number | null, range: boolean) {
  if (range && anchor !== null) {
    const start = Math.min(anchor, value), end = Math.max(anchor, value);
    return [...new Set([...values, ...Array.from({ length: end - start + 1 }, (_, i) => start + i)])].sort((a, b) => a - b);
  }
  return values.includes(value) ? values.filter(item => item !== value) : [...values, value].sort((a, b) => a - b);
}

type Props = { studyId: string; revision: number; progress: AnnotationProgress | null; time: number; slice: number; frames: number; depth: number;
  timeSpacing: number; disabled: boolean; playing: boolean; failure: string; onTime: (time: number) => void; onSlice: (slice: number) => void;
  lockedSlices: [number, number][]; locksEnabled: boolean; activityEnabled: boolean; lockDisabled: boolean;
  onSliceLocks: (times: number[], slices: number[], locked: boolean) => void; onSelectionMode: () => void };
export default function AnnotationTimeline({ studyId, revision, progress, time, slice, frames, depth, timeSpacing, disabled, playing, failure,
  onTime, onSlice, lockedSlices, locksEnabled, activityEnabled, lockDisabled, onSliceLocks, onSelectionMode }: Props) {
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem('kprism-annotation-history-collapsed') === 'true'; } catch { return false; }
  });
  const [selectMode, setSelectMode] = useState(false), [activityOpen, setActivityOpen] = useState(false);
  const [selectedTimes, setSelectedTimes] = useState<number[]>([]), [selectedSlices, setSelectedSlices] = useState<number[]>([]);
  const timeAnchor = useRef<number | null>(null), sliceAnchor = useRef<number | null>(null);
  const bodyId = useId();
  const locked = useMemo(() => new Set(lockedSlices.map(([t, z]) => `${t}:${z}`)), [lockedSlices]);
  const currentLocked = locked.has(`${time}:${slice}`);
  const selectedKeys = selectedTimes.flatMap(t => selectedSlices.map(z => `${t}:${z}`));
  useEffect(() => { if (playing) { setActivityOpen(false); setSelectMode(false); } }, [playing]);
  useEffect(() => {
    try { localStorage.setItem('kprism-annotation-history-collapsed', String(collapsed)); } catch { /* Storage may be unavailable. */ }
  }, [collapsed]);
  const frameStrip = useRef<HTMLDivElement>(null), sliceStrip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const strip = frameStrip.current, button = strip?.children[time] as HTMLElement | undefined;
    if (strip && button && (button.offsetLeft < strip.scrollLeft || button.offsetLeft + button.offsetWidth > strip.scrollLeft + strip.clientWidth))
      strip.scrollLeft = button.offsetLeft - strip.clientWidth / 2;
  }, [time, !!progress, collapsed]);
  useEffect(() => {
    const strip = sliceStrip.current, button = strip?.children[slice] as HTMLElement | undefined;
    if (strip && button && (button.offsetLeft < strip.scrollLeft || button.offsetLeft + button.offsetWidth > strip.scrollLeft + strip.clientWidth))
      strip.scrollLeft = button.offsetLeft - strip.clientWidth / 2;
  }, [slice, !!progress, collapsed]);

  const summary = useMemo(() => Array.from({ length: frames }, (_, frame) => {
    let masks = 0, edits = 0, protectedLayers = 0, latest: string | null = null;
    for (let layer = 0; layer < depth; layer++) {
      const index = frame * depth + layer, date = progress?.edited_at[index];
      if (progress?.mask_labels[index]) masks++;
      if (progress?.edited_labels[index]) edits++;
      if (locked.has(`${frame}:${layer}`)) protectedLayers++;
      if (date && (!latest || date > latest)) latest = date;
    }
    return { masks, edits, protectedLayers, latest };
  }), [progress, frames, depth, locked]);
  const index = time * depth + slice, edits = progress?.edited_labels[index], masks = progress?.mask_labels[index];
  function toggleSelectMode() {
    onSelectionMode(); setActivityOpen(false);
    if (!selectMode) {
      setSelectedTimes([time]); setSelectedSlices([slice]); timeAnchor.current = time; sliceAnchor.current = slice;
    }
    setSelectMode(value => !value);
  }
  function chooseFrame(frame: number, range: boolean) {
    if (!selectMode) { onTime(frame); return; }
    if (lockDisabled) return;
    setSelectedTimes(previous => choose(previous, frame, timeAnchor.current, range));
    if (!range) timeAnchor.current = frame;
  }
  function chooseSlice(layer: number, range: boolean) {
    if (!selectMode) { onSlice(layer); return; }
    if (lockDisabled) return;
    setSelectedSlices(previous => choose(previous, layer, sliceAnchor.current, range));
    if (!range) sliceAnchor.current = layer;
  }
  return <div className={`annotation-timeline ${collapsed ? 'collapsed' : ''}`} data-testid="annotation-timeline"
    onKeyDown={event => {
      if (event.code === 'Space' || event.key.startsWith('Arrow')) event.stopPropagation();
      if (event.key === 'Escape' && selectMode) { setSelectMode(false); event.stopPropagation(); }
    }}>
    <div className="annotation-legend">
      <button className="annotation-toggle" aria-expanded={!collapsed} aria-controls={bodyId} aria-label="Toggle edit history"
        title={collapsed ? 'Expand edit history' : 'Collapse edit history'} onClick={() => setCollapsed(value => !value)}>
        <ChevronRight size={15} className={collapsed ? '' : 'expanded'} /><span className="annotation-title">EDIT HISTORY</span>
      </button>
      {!collapsed && <><span className="progress-key masked">Mask present</span><span className="progress-key edited">Edited</span><span className="progress-key empty">Empty</span>
        {locksEnabled && <span className="annotation-badge-key"><LockKeyhole size={12} /> Locked</span>}</>}
      {progress && <span className="annotation-count" data-testid="annotation-summary">Edited frames {summary.filter(frame => frame.edits).length}/{frames}{locksEnabled && ` · Locked slices ${locked.size}/${frames * depth}`}</span>}
    </div>
    <div id={bodyId} hidden={collapsed}>
    {!collapsed && <>
    <div className="frame-lock-tools">
      {locksEnabled && <><button className={selectMode ? 'active' : ''} aria-label="Select frames and layers to lock" aria-pressed={selectMode} disabled={disabled}
        onClick={toggleSelectMode}><MousePointer2 size={14} />{selectMode ? 'Done selecting' : 'Select / Lock'}</button>
      <span className={currentLocked ? 'current-frame-lock locked' : 'current-frame-lock'} data-testid="current-slice-lock">
        {currentLocked ? <LockKeyhole size={13} /> : <LockKeyholeOpen size={13} />}Frame {time + 1} · Layer {slice + 1}: {currentLocked ? 'locked' : 'unlocked'}</span></>}
      {activityEnabled && <ActivityHistory id={studyId} revision={revision} time={time} slice={slice} open={activityOpen}
        onOpen={value => { setActivityOpen(value); if (value) { onSelectionMode(); setSelectMode(false); } }} />}
      {locksEnabled && !selectMode && <button className="quick-frame-lock" disabled={lockDisabled} onClick={() => onSliceLocks([time], [slice], !currentLocked)}
        aria-label={currentLocked ? 'Unlock current slice' : 'Lock current slice'}>{currentLocked ? 'Unlock slice' : 'Lock slice'}</button>}
    </div>
    {locksEnabled && selectMode && <div className="frame-lock-selection" role="group" aria-label="Slice lock selection">
      <div className="frame-lock-selection-actions">
        <button disabled={lockDisabled} onClick={() => { setSelectedTimes([time]); setSelectedSlices([slice]); timeAnchor.current = time; sliceAnchor.current = slice; }}>Select current slice</button>
        <button disabled={lockDisabled} onClick={() => { setSelectedTimes(Array.from({ length: frames }, (_, i) => i)); timeAnchor.current = 0; }}>All frames</button>
        <button disabled={lockDisabled} onClick={() => { setSelectedSlices(Array.from({ length: depth }, (_, i) => i)); sliceAnchor.current = 0; }}>All layers</button>
        <button disabled={lockDisabled || (!selectedTimes.length && !selectedSlices.length)} onClick={() => { setSelectedTimes([]); setSelectedSlices([]); timeAnchor.current = null; sliceAnchor.current = null; }}>Clear selection</button>
        <button disabled={lockDisabled || !selectedKeys.some(key => !locked.has(key))} onClick={() => onSliceLocks(selectedTimes, selectedSlices, true)} data-testid="lock-selected-slices"><LockKeyhole size={13} />Lock selected</button>
        <button disabled={lockDisabled || !selectedKeys.some(key => locked.has(key))} onClick={() => onSliceLocks(selectedTimes, selectedSlices, false)} data-testid="unlock-selected-slices"><LockKeyholeOpen size={13} />Unlock selected</button>
      </div>
      <span data-testid="slice-lock-scope">Frames {ranges(selectedTimes) || 'none'} · Layers {ranges(selectedSlices) || 'none'} · RV + MYO + LV ({selectedKeys.length} {selectedKeys.length === 1 ? 'slice' : 'slices'})</span>
      <small>Select frame and layer numbers below; Shift + click selects a range. Each selected frame is paired with each selected layer. Use All layers only when you want to protect a complete frame. Empty masks can also be locked.</small>
    </div>}
    {progress ? <>
    {(frames > 1 || selectMode) && <div className="annotation-track"><span className="navigator-label">FRAMES</span><div className="annotation-markers" ref={frameStrip}>
      {summary.map((frame, index) => <button key={index} className={`annotation-marker ${state(frame.masks, frame.edits)} ${index === time ? 'current' : ''} ${selectMode && selectedTimes.includes(index) ? 'lock-selected' : ''}`}
        data-testid={`annotation-frame-${index}`} data-state={state(frame.masks, frame.edits)} data-locked={frame.protectedLayers === depth} data-locked-layers={frame.protectedLayers}
        aria-pressed={selectMode ? selectedTimes.includes(index) : index === time}
        aria-label={`${selectMode ? 'Select' : 'Go to'} frame ${index + 1}${frame.protectedLayers ? `, ${frame.protectedLayers} of ${depth} layers locked` : ''}`} disabled={selectMode ? lockDisabled : disabled} onClick={event => chooseFrame(index, event.shiftKey)}
        title={`Frame ${index + 1}${timeSpacing > 0 ? ` · ${(index * timeSpacing).toFixed(3)} s` : ''} · ${frame.edits}/${depth} layers edited · ${frame.masks}/${depth} layers with masks\n${stamp(frame.latest)}${frame.protectedLayers ? `\nLocked: ${frame.protectedLayers}/${depth} layers · RV + MYO + LV` : ''}${selectMode ? '\nClick to select; Shift + click selects a range' : ''}`}>
        {index + 1}{frame.protectedLayers > 0 && <><LockKeyhole size={11} aria-hidden="true" /><span className="annotation-lock-count">{frame.protectedLayers}/{depth}</span></>}</button>)}
    </div></div>}
    <div className="annotation-track"><span className="navigator-label">LAYERS</span><div className="annotation-markers" ref={sliceStrip}>
      {Array.from({ length: depth }, (_, layer) => {
        const index = time * depth + layer, edited = progress.edited_labels[index], mask = progress.mask_labels[index], protectedSlice = locked.has(`${time}:${layer}`);
        return <button key={layer} className={`annotation-marker ${state(mask, edited)} ${layer === slice ? 'current' : ''} ${selectMode && selectedSlices.includes(layer) ? 'lock-selected' : ''}`}
          data-testid={`annotation-layer-${layer}`} data-state={state(mask, edited)} data-locked={protectedSlice} aria-pressed={selectMode ? selectedSlices.includes(layer) : layer === slice}
          aria-label={`${selectMode ? 'Select' : 'Go to'} layer ${layer + 1}${protectedSlice ? ', locked in current frame' : ''}`} disabled={selectMode ? lockDisabled : disabled} onClick={event => chooseSlice(layer, event.shiftKey)}
          title={`Frame ${time + 1}, layer ${layer + 1}\n${edited ? `Edited: ${organs(edited)}` : 'No manual edits recorded'}\n${mask ? `Masks: ${organs(mask)}` : 'Mask empty'}\n${stamp(progress.edited_at[index])}${protectedSlice ? '\nThis slice is locked for RV + MYO + LV' : ''}${selectMode ? '\nClick to select; Shift + click selects a range' : ''}`}>
          {layer + 1}{protectedSlice && <LockKeyhole size={11} aria-hidden="true" />}</button>;
      })}
    </div></div>
    <div className="annotation-details" data-testid="annotation-current-layer">Frame {time + 1} · Layer {slice + 1}
      {frames > 1 && timeSpacing > 0 && ` · ${(time * timeSpacing).toFixed(3)} s`} · {edits ? `Edited ${organs(edits)} · ${stamp(progress.edited_at[index])}` : 'No manual edits recorded'}
      {currentLocked && ' · Slice locked'}{!masks && ' · mask empty'}{!progress.history_complete && ' · Earlier edit history is incomplete'}
    </div>
    {failure && <div className="note-error" role="status">Could not refresh edit history: {failure}</div>}
    </> : <div className="annotation-pending" role="status">{failure || 'Loading annotation history…'}</div>}
    </>}
    </div>
  </div>;
}
