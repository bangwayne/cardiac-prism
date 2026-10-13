import type { GPUDevice } from './types';

export type GPUSelection = { enabled: boolean; selected: number[]; onEnabled: (enabled: boolean) => void; onSelected: (ids: number[]) => void };
export default function GPUSelector({ gpus, selection, disabled, scope }: {
  gpus: GPUDevice[]; selection: GPUSelection; disabled: boolean; scope: string;
}) {
  return <div className="gpu-selector">
    <label className="checkbox-label"><input type="checkbox" checked={selection.enabled} disabled={disabled}
      onChange={event => selection.onEnabled(event.target.checked)} aria-label={`Choose GPUs for ${scope}`} /><span>Choose GPUs</span></label>
    {selection.enabled ? <>
      <div className="gpu-options" role="group" aria-label={`GPUs for ${scope}`}>
        {gpus.map(gpu => <label className={`gpu-option ${gpu.available ? '' : 'unavailable'}`} key={gpu.uuid}>
          <input type="checkbox" aria-label={`GPU ${gpu.id}`} checked={selection.selected.includes(gpu.id)} disabled={disabled || !gpu.available}
            onChange={event => selection.onSelected(event.target.checked ? [...selection.selected, gpu.id] : selection.selected.filter(id => id !== gpu.id))} />
          <span><strong>GPU {gpu.id}</strong><small>{gpu.name}</small></span>
          <span className="gpu-memory">{(gpu.free_mib / 1024).toFixed(1)} GB free{!gpu.available && <small>Unavailable</small>}</span>
        </label>)}
      </div>
      <p className="muted-help">{selection.selected.length > 1 ? `${selection.selected.length} GPUs process different planes in parallel.` : selection.selected.length ? 'One GPU selected. Select more available GPUs to run in parallel.' : 'Select at least one available GPU.'}</p>
    </> : <p className="muted-help">Uses the current CPU / GPU inference device.</p>}
  </div>;
}
