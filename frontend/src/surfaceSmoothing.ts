type Geometry = { points: Float32Array; normals: Float32Array };
type Task = { id: number; key: string; points: Float32Array; polys: Uint32Array;
  resolve: (result: Geometry) => void; reject: (error: Error) => void; cancelled: boolean };
let worker: Worker | null = null, active: Task | null = null, sequence = 0;
const waiting: Task[] = [];
function next() {
  if (active || !waiting.length) return;
  active = waiting.shift()!;
  if (active.cancelled) { active = null; next(); return; }
  try {
    if (!worker) {
      worker = new Worker(new URL('./surfaceSmoothing.worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = event => {
        const task = active; if (!task || event.data.id !== task.id) return;
        active = null;
        if (!task.cancelled) {
          if (event.data.error) task.reject(new Error(event.data.error)); else task.resolve(event.data);
        }
        next();
      };
      worker.onerror = event => {
        active?.reject(new Error(event.message)); active = null;
        worker?.terminate(); worker = null;
        for (const task of waiting.splice(0)) task.reject(new Error(event.message));
      };
    }
    worker.postMessage({ id: active.id, points: active.points, polys: active.polys });
  } catch (error) { active.reject(error instanceof Error ? error : new Error(String(error))); active = null; next(); }
}
export function smoothSurface(key: string, points: Float32Array, polys: Uint32Array) {
  // Prefer the current refinement over waiting work from retired surfaces.
  for (const task of waiting) if (task.key === key) { task.cancelled = true; task.reject(new Error('Superseded surface')); }
  let task: Task;
  const promise = new Promise<Geometry>((resolve, reject) => {
    task = { id: ++sequence, key, points, polys, resolve, reject, cancelled: false };
    waiting.unshift(task); next();
  });
  return { promise, cancel: () => { task.cancelled = true; task.reject(new Error('Retired surface')); } };
}
