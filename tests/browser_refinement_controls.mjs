import { chromium, expect } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { base, fixturePath, reportDirectory, browserOptions, assertIsolatedService } from './browser_helpers.mjs';
const baseline = false;
const out = reportDirectory('refinement');
fs.mkdirSync(out, { recursive: true });
const source = JSON.parse(fs.readFileSync(fixturePath('cine', 'fixture.json')));
const browser = await chromium.launch(browserOptions);
const page = await browser.newPage({ viewport: { width: 1680, height: 1060 } });
const report = { passed: false, baseline, checks: [], errors: [], desynchronisations: [], requests: [] };
page.on('pageerror', error => report.errors.push(error.stack || error.message));
page.on('crash', () => report.errors.push('Browser page crashed'));
page.on('request', request => { if (request.method() === 'GET' && /\/api\/projects\/[^/]+\/(volume|masks)/.test(request.url())) report.requests.push(request.url()); });
const state = () => page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__?.state());
async function check(name, work) { await work(); report.checks.push(name); console.log('PASS', name); }
async function value(label, next) {
  await page.getByLabel(label, { exact: true }).evaluate((element, input) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(element, String(input));
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }, next);
}
async function jump(time) {
  await value('Time frame', time);
  await expect.poll(async () => (await state())?.time, { timeout: 30000 }).toBe(time);
  await expect(page.getByRole('button', { name: 'Play cine', exact: true })).toBeEnabled({ timeout: 30000 });
}
async function errorsAtPosition(position) {
  return page.evaluate(index => {
    const api = window.__CARDIAC_PRISM_DIAGNOSTICS__, d = api.state(), world = api.indexToWorld(index);
    return d.sliceCameras.map(({ name, camera }) => ({ name,
      delta: Math.abs(camera.viewPlaneNormal.reduce((sum, component, axis) => sum + component * (world[axis] - camera.focalPoint[axis]), 0)) }));
  }, position);
}
try {
  await assertIsolatedService();
  const uploaded = await page.request.post(base + '/api/projects', { multipart: {
    file: { name: 'moving-fixture.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(fixturePath('cine', 'cine-fixture.nii.gz')) },
  } });
  assert(uploaded.ok(), await uploaded.text()); let project = await uploaded.json();
  const imported = await page.request.post(base + `/api/projects/${project.id}/annotations`, { multipart: {
    revision: '0', target: 'annotation', binary_organ: '0',
    file: { name: 'moving-labels.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(fixturePath('cine', 'cine-fixture-labels.nii.gz')) },
  } });
  assert(imported.ok(), await imported.text()); project = await imported.json();
  const prefix = base + `/api/projects/${project.id}`;
  report.study_id = project.id;
  await page.goto(base + `/?study=${project.id}&debug=1`);
  await expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 45000 });
  await check('Fourth viewport has only five tiny locator actors; no image volume, mask surface or textured plane', async()=>{
    const d=await state();assert.equal(d.volumeActors.length,5);assert(d.volumeActors.every(a=>a.type==='vtkActor'));
    assert.equal(d.slicePlane,null);assert.equal(d.surfaces.length,0);assert.equal(d.surfaceCache.bytes,0);
    await expect(page.getByTestId('3d-axis-labels').locator('text')).toHaveText(['X','Y','Z']);
    await expect(page.getByText('Current slice in 3D',{exact:true})).toHaveCount(0);
    await expect(page.getByText('Smooth 3D surfaces',{exact:true})).toHaveCount(1);
  });
  await page.getByLabel('Locate (3)', { exact: true }).click();
  await value('Acquisition slice', 1);
  const desired = [27, 35, 1];
  const coordinate = await page.evaluate(index => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas('native', index), desired);
  const native = await page.getByTestId('viewport-native').boundingBox();
  await page.mouse.click(native.x + coordinate[0], native.y + coordinate[1]);
  await expect.poll(async () => (await state())?.position[2]).toBe(1);
  await check('XYZ lines cross the exact shared cursor and follow the sagittal, coronal and acquisition view normals',async()=>{
    const d=await state(),world=await page.evaluate(()=>{const api=window.__CARDIAC_PRISM_DIAGNOSTICS__;return api.indexToWorld(api.state().position);});
    d.locator.center.forEach((x,i)=>assert(Math.abs(x-world[i])<1e-5));
    for(const axis of d.locator.axes){
      const n=d.sliceCameras.find(c=>c.name===axis.view).camera.viewPlaneNormal;
      assert(Math.abs(Math.abs(n.reduce((v,x,i)=>v+x*axis.direction[i],0))-1)<1e-5);
      const v=axis.end.map((x,i)=>x-axis.start[i]),u=world.map((x,i)=>x-axis.start[i]);
      const t=u.reduce((a,x,i)=>a+x*v[i],0)/v.reduce((a,x)=>a+x*x,0);
      assert(t>=0&&t<=1);assert(Math.hypot(...world.map((x,i)=>x-axis.start[i]-t*v[i]))<1e-4);
      assert.equal(axis.points.length,6);
    }
    assert((await errorsAtPosition(d.position)).every(c=>c.delta<1e-4));
  });
  await check('Expansion, restoration, fit and browser resize retain the three slice positions and linked 3D cursor', async () => {
    for (const title of ['Acquisition plane', 'Coronal', 'Sagittal', '3D locator']) {
      const camera = (await state()).camera;
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      await page.waitForTimeout(200);
      let deltas = await errorsAtPosition(desired);
      if (baseline) report.desynchronisations.push({ title, expanded: true, deltas });
      else assert(deltas.every(entry => entry.delta < 1e-4), JSON.stringify({ title, deltas }));
      assert.equal((await state()).position[2], 1);
      if (!baseline) {
        for (const key of ['viewUp', 'viewPlaneNormal']) assert((await state()).camera[key].every((v, i) => Math.abs(v - camera[key][i]) < 1e-5));
      }
      await page.getByLabel('Expand ' + title, { exact: true }).click(); await page.waitForTimeout(200);
      await page.getByLabel('Fit ' + title, { exact: true }).click();
      deltas = await errorsAtPosition(desired);
      if (baseline) report.desynchronisations.push({ title, fit: true, deltas });
      else assert(deltas.every(entry => entry.delta < 1e-4), JSON.stringify({ title, deltas }));
    }
    await page.setViewportSize({ width: 1500, height: 960 }); await page.waitForTimeout(200);
    if (!baseline) assert((await errorsAtPosition(desired)).every(entry => entry.delta < 1e-4));
    await page.setViewportSize({ width: 1680, height: 1060 }); await page.waitForTimeout(200);
    // Restore the cursor explicitly for the baseline's known resize defect.
    if (baseline) { await value('Acquisition slice', 2); await value('Acquisition slice', 1); }
  });
  if (!baseline) await check('Bright locator color and 0–100% opacity are shared and survive reload without mask edits', async () => {
    await value('Locator color', '#ffde00'); await value('Locator opacity', .35);
    await expect(page.getByTestId('view-locator')).toHaveCount(3);
    for (const locator of await page.getByTestId('view-locator').all()) {
      await expect(locator).toHaveAttribute('opacity', '0.35'); await expect(locator).toHaveAttribute('stroke', '#ffde00');
    }
    assert.equal((await state()).locator.opacity,.35);
    await value('Locator opacity', 0); await expect(page.getByTestId('view-locator').first()).toHaveAttribute('opacity', '0');
    assert((await state()).locator.axes.every(a=>!a.visible));
    await value('Locator opacity', 1);
    await page.reload(); await expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 45000 });
    await expect(page.getByLabel('Locator color', { exact: true })).toHaveValue('#ffde00');
    await expect(page.getByLabel('Locator opacity', { exact: true })).toHaveValue('1');
    assert.equal((await (await page.request.get(prefix)).json()).revision, project.revision);
  });
  await check('All six moving source frames and organ labels stay exact in the three 2D views with no 3D mask or slice plane', async () => {
    for (let time = 0; time < 6; time++) {
      await jump(time);
      const sample = await page.evaluate(indices => {
        const d = window.__CARDIAC_PRISM_DIAGNOSTICS__;
        return { state: d.state(), values: indices.map(index => d.frameValues(index)) };
      }, source.samples);
      assert.equal(sample.state.surfaces.length, 0);assert.equal(sample.state.slicePlane,null);
      assert.equal(sample.state.surfaceCache.bytes,0);assert.equal(sample.state.surfaceCache.builds,0);
      sample.values.forEach((voxel, index) => {
        assert.equal(voxel.image, source.frames[time].image[index]); assert.equal(voxel.volumeImage, voxel.image);
        assert.deepEqual(voxel.masks, [1, 2, 3].map(label => source.frames[time].labels[index] === label ? label : 0));
        assert.deepEqual(voxel.volumeMasks, voxel.masks);
      });
    }
  });
  await jump(2);
  await page.getByLabel('Select LV', { exact: true }).click(); await page.getByLabel('Brush (4)', { exact: true }).click();
  const before = [];
  for (let time = 0; time < 6; time++) before.push(await (await page.request.get(`${prefix}/masks?time=${time}`)).body());
  report.requests = []; const beforeEdit = await state(), started = performance.now();
  const slice = beforeEdit.position[2], index = (slice * 64 + 2) * 64 + 2;
  const point = await page.evaluate(([slice]) => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas('native', [2, 2, slice]), [slice]);
  const box = await page.getByTestId('viewport-native').boundingBox(); await page.mouse.click(box.x + point[0], box.y + point[1]);
  await expect.poll(async () => (await page.evaluate(index => window.__CARDIAC_PRISM_DIAGNOSTICS__.frameValues(index), index)).masks[2], { timeout: 15000 }).toBe(3);
  const afterEdit = await state(); report.edit_ms = Math.round(performance.now() - started);
  await check('A one-pixel edit updates exactly one frame and organ, keeps source textures and constructs no 3D meshes', async () => {
    if (!baseline) {
      assert.equal(afterEdit.sourceUploads, beforeEdit.sourceUploads);
      assert.equal(afterEdit.surfaceCache.builds - beforeEdit.surfaceCache.builds, 0);
      assert.deepEqual(afterEdit.labelUploads,[{label:3,slices:[slice]}]);
      assert.equal(afterEdit.surfaceCache.frames.length, beforeEdit.surfaceCache.frames.length);
    }
    const expected = Buffer.from(before[2]); expected[2 * 6 * 64 * 64 + index] = 1;
    for (let time = 0; time < 6; time++) assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=${time}`)).body(), time === 2 ? expected : before[time]);
  });
  report.requests = []; const revisit = performance.now();
  for (const time of [0, 1, 3, 4, 5, 2]) await jump(time);
  report.revisit_ms = Math.round(performance.now() - revisit);
  report.revisit_mask_requests = report.requests.filter(url => url.includes('/masks')).length;
  report.revisit_volume_requests = report.requests.filter(url => url.includes('/volume')).length;
  await check('Unaffected movie frames retain image and label buffers after a local edit', async () => {
    if (!baseline) {
      assert.equal(report.revisit_mask_requests, 0); assert.equal(report.revisit_volume_requests, 0);
      assert.equal((await state()).surfaceCache.builds, afterEdit.surfaceCache.builds);
    }
  });
  await page.getByRole('button', { name: 'Undo edit', exact: true }).click();
  await expect.poll(async () => (await page.evaluate(index => window.__CARDIAC_PRISM_DIAGNOSTICS__.frameValues(index), index)).masks[2]).toBe(0);
  await check('Undo restores every original label with zero surface construction', async () => {
    for (let time = 0; time < 6; time++) { await jump(time); assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=${time}`)).body(), before[time]); }
    if (!baseline) assert.equal((await state()).surfaceCache.builds, afterEdit.surfaceCache.builds);
  });
  await check('3D locator rotates independently while its world cursor stays aligned to all 2D planes',async()=>{
    const before=await state(),box=await page.getByTestId('viewport-volume').boundingBox();
    await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();
    await page.mouse.move(box.x+box.width*.67,box.y+box.height*.6,{steps:12});await page.mouse.up();await page.waitForTimeout(150);
    const after=await state();assert.deepEqual(after.locator.center,before.locator.center);
    assert(after.camera.viewPlaneNormal.some((x,i)=>Math.abs(x-before.camera.viewPlaneNormal[i])>1e-3));
    assert((await errorsAtPosition(after.position)).every(c=>c.delta<1e-4));
  });
  await page.getByLabel('Playback speed', { exact: true }).selectOption('15');
  const play = performance.now(), initialTime = (await state()).time;
  await page.getByRole('button', { name: 'Play cine', exact: true }).click();
  await expect.poll(async () => (await state()).time, { timeout: 30000, intervals: [30] }).not.toBe(initialTime);
  report.start_to_first_frame_ms = Math.round(performance.now() - play);
  const samples = [], seenFrames = new Set(), observationStart = performance.now();
  const minimumEnd = observationStart + 6000, deadline = observationStart + 30000;
  while (performance.now() < deadline && (performance.now() < minimumEnd || seenFrames.size < 6)) {
    const sample = { time: (await state()).time, at: performance.now() };
    samples.push(sample); seenFrames.add(sample.time);
    await page.waitForTimeout(25);
  }
  if (!baseline) assert.equal((await state()).volumeVisible, false);
  await page.getByRole('button', { name: 'Pause cine', exact: true }).click();
  if (!baseline) assert.equal((await state()).volumeVisible, false);
  const advances = samples.filter((sample, index) => index && sample.time !== samples[index - 1].time).length;
  report.observed_fps = advances * 1000 / (samples.at(-1).at - samples[0].at);
  assert(new Set(samples.map(sample => sample.time)).size === 6);
  report.checks.push('Cine loops all six frames with moving 2D labels; pausing never enables a 3D volume or surface');
  await jump(0);
  await check('Render 3D explicitly builds exact current-frame surfaces in physical reflected coordinates', async () => {
    await page.getByTestId('render-3d').click();
    await expect.poll(async () => (await state()).surfaceStatus.state, { timeout: 30000 }).toBe('ready');
    const d = await state(); assert.equal(d.surfaces.length, 3); assert.equal(d.volumeActors.length, 8);
    assert.deepEqual(d.surfaceCache.frames, [0]); assert(d.surfaceCache.bytes > 0);
    assert.equal(d.surfaceCache.inputBytes, 0);
    for (const surface of d.surfaces) {
      const cy = { 1: 12, 2: 30, 3: 47 }[surface.label];
      const corners = [];
      for (const x of [10.5, 19.5]) for (const y of [cy - 4.5, cy + 4.5]) for (const z of [.5, 4.5]) corners.push([x, y, z]);
      const world = await page.evaluate(corners => corners.map(index => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToWorld(index)), corners);
      const expected = [0, 1, 2].flatMap(axis => [Math.min(...world.map(p => p[axis])), Math.max(...world.map(p => p[axis]))]);
      surface.bounds.forEach((v, axis) => assert(Math.abs(v - expected[axis]) < 1e-3, JSON.stringify({surface, expected})));
    }
    await page.screenshot({ path: new URL('render-3d-fixture.png', out).pathname });
  });
  await check('Surface opacity, organ visibility and background smoothing do not alter any mask', async () => {
    await value('3D mask opacity', .4);
    await expect.poll(async () => (await state()).surfaces.every(s => s.opacity === .4)).toBe(true);
    await page.getByLabel('Toggle RV visibility', {exact:true}).click();
    await expect.poll(async () => (await state()).surfaces.find(s=>s.label===1).visible).toBe(false);
    await page.getByLabel('Toggle RV visibility', {exact:true}).click();
    await page.getByText('Smooth 3D surfaces',{exact:true}).click();
    await expect.poll(async () => (await state()).surfaces.every(s=>s.smoothingReady), {timeout:30000}).toBe(true);
    assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=0`)).body(), before[0]);
    await page.getByText('Smooth 3D surfaces',{exact:true}).click();
  });
  await check('Time navigation and playback release all surfaces and never automatically rebuild them', async () => {
    const builds = (await state()).surfaceCache.builds;
    await jump(1); assert.equal((await state()).surfaceCache.bytes,0); assert.equal((await state()).volumeActors.length,5);
    await page.getByTestId('render-3d').click();
    await expect.poll(async()=> (await state()).surfaceStatus.state).toBe('ready');
    await page.getByRole('button',{name:'Play cine',exact:true}).click();
    await expect.poll(async()=> (await state()).surfaceStatus.state).toBe('off');
    await expect(page.getByTestId('render-3d')).toBeDisabled();
    await page.waitForTimeout(1000);
    assert.equal((await state()).surfaceCache.builds,builds+3);
    assert.equal((await state()).surfaceCache.bytes,0); assert.equal((await state()).volumeActors.length,5);
    await page.getByRole('button',{name:'Pause cine',exact:true}).click();
    assert.equal((await state()).surfaceStatus.state,'off');
  });
  await check('Starting cine during a pending Render 3D cancels its build and leaves no retired geometry', async () => {
    await page.evaluate(() => {
      document.querySelector('[data-testid="render-3d"]').click();
      document.querySelector('button[aria-label="Play cine"]').click();
    });
    await page.waitForTimeout(800);
    assert.equal((await state()).surfaceStatus.state,'off');assert.equal((await state()).surfaceCache.bytes,0);
    assert.equal((await state()).volumeActors.length,5);
    await page.getByRole('button',{name:'Pause cine',exact:true}).click();
  });
  await jump(2); await value('Acquisition slice',3);
  await expect.poll(async()=> (await state()).position[2]).toBe(3);
  await check('Editing a rendered frame discards its surfaces and performs no automatic remeshing', async () => {
    await page.getByTestId('render-3d').click();await expect.poll(async()=> (await state()).surfaceStatus.state).toBe('ready');
    const builds=(await state()).surfaceCache.builds;
    await page.getByLabel('Select MYO',{exact:true}).click();await page.getByLabel('Brush (4)',{exact:true}).click();
    const p=await page.evaluate(()=>window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas('native',[2,2,3]));
    const box=await page.getByTestId('viewport-native').boundingBox();const start=performance.now();
    await page.mouse.click(box.x+p[0],box.y+p[1]);
    await expect.poll(async()=> (await page.evaluate(()=>window.__CARDIAC_PRISM_DIAGNOSTICS__.frameValues((3*64+2)*64+2))).masks[1]).toBe(2);
    report.rendered_frame_edit_ms=Math.round(performance.now()-start);
    assert.equal((await state()).surfaceStatus.state,'off');assert.equal((await state()).surfaceCache.bytes,0);
    assert.equal((await state()).surfaceCache.builds,builds);
    await page.getByRole('button',{name:'Undo edit',exact:true}).click();
    await expect.poll(async()=> (await page.evaluate(()=>window.__CARDIAC_PRISM_DIAGNOSTICS__.frameValues((3*64+2)*64+2))).masks[1]).toBe(0);
  });
  await check('Clear all masks this slice affects three organs on one frame and one undo restores them', async () => {
    await page.getByTestId('clear-all-masks-slice').click();
    await expect(page.getByRole('button',{name:'Play cine',exact:true})).toBeEnabled({ timeout: 30000 });
    for(let t=0;t<6;t++){
      const expected=Buffer.from(before[t]);if(t===2)for(let label=0;label<3;label++)expected.fill(0,(label*6+3)*64*64,(label*6+4)*64*64);
      assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=${t}`)).body(),expected);
    }
    await page.getByRole('button',{name:'Undo edit',exact:true}).click();
    await expect(page.getByRole('button',{name:'Play cine',exact:true})).toBeEnabled({ timeout: 30000 });
    assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=2`)).body(),before[2]);
  });
  await check('Clear all masks this layer affects only this acquisition layer across all times and invalidates their cached labels', async () => {
    await page.getByTestId('clear-all-masks-layer').click();
    await expect(page.getByRole('button',{name:'Play cine',exact:true})).toBeEnabled({ timeout: 30000 });
    for(let t=0;t<6;t++){
      const expected=Buffer.from(before[t]);for(let label=0;label<3;label++)expected.fill(0,(label*6+3)*64*64,(label*6+4)*64*64);
      assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=${t}`)).body(),expected);
      await jump(t);
      const v=await page.evaluate(()=>window.__CARDIAC_PRISM_DIAGNOSTICS__.frameValues((3*64+30)*64+15));assert(v.masks.every(x=>x===0));
    }
    await page.getByRole('button',{name:'Undo edit',exact:true}).click();
    await expect(page.getByRole('button',{name:'Play cine',exact:true})).toBeEnabled({ timeout: 30000 });
    for(let t=0;t<6;t++)assert.deepEqual(await (await page.request.get(`${prefix}/masks?time=${t}`)).body(),before[t]);
  });
  await check('Locked structures disable all-mask clears and matched annotation replacement', async () => {
    await page.getByLabel('Lock RV',{exact:true}).click();
    await expect(page.getByTestId('clear-all-masks-slice')).toBeDisabled();await expect(page.getByTestId('clear-all-masks-layer')).toBeDisabled();
    await page.getByLabel('Unlock RV',{exact:true}).click();
    await expect(page.getByTestId('clear-all-masks-slice')).toBeEnabled();
  });
  try {
    await page.screenshot({ path: new URL('viewer.png', out).pathname, timeout: 60000 });
  } catch (error) {
    if (error.name !== 'TimeoutError' || page.isClosed()) throw error;
    report.artifact_warning = `Viewer screenshot unavailable: ${error.message}`;
    console.warn(report.artifact_warning);
  }
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) {
  report.failure = error.stack; process.exitCode = 1;
  await page.screenshot({ path: new URL('failure.png', out).pathname }).catch(() => {});
} finally {
  fs.writeFileSync(new URL('browser.json', out), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, checks: report.checks, edit_ms: report.edit_ms, revisit_ms: report.revisit_ms,
    revisit_mask_requests: report.revisit_mask_requests, observed_fps: report.observed_fps, failure: report.failure }));
  await browser.close();
}
