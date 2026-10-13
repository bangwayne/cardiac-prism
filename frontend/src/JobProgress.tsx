import type { Job } from './types';

export default function JobProgress({ job, elapsed, pollingWarning }: {
  job: Job; elapsed: number; pollingWarning: string;
}) {
  const gpuIds = job.gpu_ids ?? [], ready = new Set(job.ready_gpu_ids ?? []);
  const preparing = new Set(job.preparing_gpu_ids ?? []);
  const batch = ['all_frames', 'all_layers', 'propagation'].includes(job.mode ?? '');
  const title = job.phase === 'applying' ? 'Applying segmentation' :
    job.state === 'queued' || job.phase === 'queued' ? 'Inference queued' :
    job.phase === 'preparing' ? (gpuIds.length ? `Preparing GPUs · ${ready.size} / ${gpuIds.length} ready` : 'Preparing inference') :
    job.mode === 'propagation' ? `Propagating layer ${job.slice + 1} in time` :
    job.mode === 'all_frames' ? 'Segmenting all time frames' :
    job.mode === 'all_layers' ? 'Segmenting all layers' : 'Refining segmentation';
  const progress = job.mode === 'propagation' ? `${job.completed_frames ?? 0} / ${job.total_frames} frames` :
    job.mode === 'all_frames' ? `${job.completed_slices ?? 0} / ${job.total_slices} planes · ${job.completed_frames ?? 0} / ${job.total_frames} frames` :
    job.mode === 'all_layers' ? `${job.completed_slices ?? 0} / ${job.total_slices} layers · frame ${job.time + 1}` : `slice ${job.slice + 1}`;
  return <span className="job-progress" role="status" aria-live="polite" data-testid="job-progress">
    <strong data-testid="job-phase">{title}</strong>
    <small data-testid="job-compute-progress">{elapsed}s · {progress}</small>
    {gpuIds.length > 0 && <small data-testid="job-gpu-progress">{gpuIds.map(id => {
      const count = job.gpu_completed?.[String(id)] ?? 0;
      return `GPU ${id}: ${job.phase === 'preparing' ? ready.has(id) ? 'ready' : preparing.has(id) ? 'loading' : 'waiting' : `${count} plane${count === 1 ? '' : 's'}`}`;
    }).join(' · ')}</small>}
    {batch && <small className="job-result-note">Masks appear together after all planes finish. Cancel discards this batch.</small>}
    {pollingWarning && <small className="job-polling-warning" data-testid="job-polling-warning">{pollingWarning}</small>}
  </span>;
}
