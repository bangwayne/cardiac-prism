import { gzipSync, gunzipSync } from 'fflate';
self.onmessage = (event: MessageEvent<{ id: number; encode: boolean; bytes: Uint8Array }>) => {
  const { id, encode, bytes } = event.data;
  try {
    const result = encode ? gzipSync(bytes, { level: 1 }) : gunzipSync(bytes);
    self.postMessage({ id, bytes: result }, { transfer: [result.buffer] });
  } catch (error) { self.postMessage({ id, error: String(error) }); }
};
