import { gzipSync, gunzipSync } from 'fflate';
let worker: Worker | null = null, sequence = 0;
const pending = new Map<number, { resolve: (bytes: Uint8Array) => void; reject: (error: Error) => void }>();
export function frameCodec(bytes: Uint8Array, encode: boolean): Promise<Uint8Array> {
  if (typeof Worker === 'undefined') return Promise.resolve(encode ? gzipSync(bytes, { level: 1 }) : gunzipSync(bytes));
  if (!worker) {
    worker = new Worker(new URL('./frameCodec.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = event => {
      const callback = pending.get(event.data.id); if (!callback) return;
      pending.delete(event.data.id);
      if (event.data.error) callback.reject(new Error(event.data.error)); else callback.resolve(event.data.bytes);
    };
    worker.onerror = event => {
      for (const callback of pending.values()) callback.reject(new Error(event.message));
      pending.clear(); worker?.terminate(); worker = null;
    };
  }
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    // Clone input: the renderer and immutable frame cache still own its buffer.
    worker!.postMessage({ id, bytes, encode });
  });
}
