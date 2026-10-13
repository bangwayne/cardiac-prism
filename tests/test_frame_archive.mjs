import assert from 'node:assert/strict';
import fs from 'node:fs';
import { build } from '../frontend/node_modules/esbuild/lib/main.js';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';
import { reportDirectory } from './browser_helpers.mjs';
const output = reportDirectory('frame-archive');
const bundle = new URL('frame-cache.mjs', output);
await build({ entryPoints: [fileURLToPath(new URL('../frontend/src/frameCache.ts', import.meta.url))],
  outfile: fileURLToPath(bundle), bundle: true, platform: 'node', format: 'esm', target: 'node22' });
const { FrameCache } = await import(pathToFileURL(fileURLToPath(bundle)).href);
let revision = 0, requests = [];
const shape = [6, 3, 5, 7], size = 105;
const images = Array.from({ length:6 }, (_, t) => Float32Array.from({ length:size }, (_, i) => Math.sin(i+t)*253.785));
const masks = Array.from({ length:6 }, (_, t) => Uint8Array.from({ length:size*3 }, (_, i) => +(i % (t+3) === 0)));
const project = (time=0) => ({ id:'archive-study', revision, shape, dimensions:[7,5,3], time, slice:1, events:[], has_reference:false });
globalThis.fetch = async path => {
  requests.push(path); const url = new URL(path,'http://localhost'), t=+url.searchParams.get('time');
  if (url.pathname.endsWith('/volume')) return new Response(images[t].buffer);
  if (url.pathname.endsWith('/masks')) return new Response(masks[t], {headers:{'X-Project-Revision':String(revision)}});
  return Response.json(project(t));
};
const cache = new FrameCache('archive-study'); cache.limit = 1500;
for (let t=0;t<6;t++) { cache.activate(t); await cache.get(project(),t,1); }
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(cache.diagnostics().archivedFrames.length,6);
assert(cache.diagnostics().frames.length<6);
requests=[];
for (let t=0;t<6;t++) {
  cache.activate(t); const frame=await cache.get(project(),t,1);
  assert.deepEqual(new Uint8Array(frame.image.buffer),new Uint8Array(images[t].buffer));
  assert.deepEqual(frame.masks,masks[t]);
}
assert.equal(requests.filter(path => /\/(masks|volume)\?/.test(path)).length,0);
const checks=['Evicted raw frames return from bounded lossless archive with every original float bit and label unchanged, without network image/mask reload'];
revision=1; masks[2][0] ^= 1; cache.invalidate(revision,[2]); requests=[];
for (let t=0;t<6;t++) { cache.activate(t); const frame=await cache.get(project(),t,1); assert.deepEqual(frame.masks,masks[t]); }
assert.equal(requests.filter(path => path.includes('/masks')).length,1);
assert.equal(requests.filter(path => path.includes('/volume')).length,0);
checks.push('Editing one archived frame reloads only its labels; unaffected archived frames and original images remain valid');
const report={passed:true,checks,cache:cache.diagnostics()};
fs.writeFileSync(new URL('archive-tests.json',output),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
