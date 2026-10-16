// Display transforms must move the rendered anatomy and annotations together,
// while edits and exports continue to address the original acquisition voxels.
import { chromium, expect } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { base, fixturePath, reportDirectory, browserOptions, assertIsolatedService } from './browser_helpers.mjs';
import { saveDiagnosticScreenshot } from './browser_artifacts.mjs';

const output = fileURLToPath(reportDirectory('view-orientation-controls'));
const python = process.env.CARDIAC_PRISM_TEST_PYTHON || 'python3';
const fixture = JSON.parse(fs.readFileSync(fixturePath('orientation', 'manifest.json'), 'utf8'))
  .find(item => item.name === 'oblique-reflected');
const labelsPath = path.join(output, 'orientation-labels.nii.gz');
execFileSync(python, ['-c',
  'import sys,SimpleITK as s,numpy as n;i=s.ReadImage(sys.argv[1]);a=n.zeros(s.GetArrayFromImage(i).shape,n.uint8);a[:,3:8,18:26,27:35]=2;o=s.GetImageFromArray(a,isVector=False);o.CopyInformation(i);s.WriteImage(o,sys.argv[2])',
  fixture.path, labelsPath]);

await assertIsolatedService();
const browser = await chromium.launch(browserOptions);
const page = await browser.newPage({ viewport: { width: 1680, height: 1060 } });
const report = { passed: false, isolated: true, checks: [], errors: [], pixels: [], prompts: [] };
page.on('pageerror', error => report.errors.push(error.stack || error.message));
page.on('crash', () => report.errors.push('Browser page crashed'));
const views = { native: { axis: 2, title: 'Acquisition plane' },
  coronal: { axis: 1, title: 'Y reconstruction' }, sagittal: { axis: 0, title: 'X reconstruction' } };
const identity = [1, 0, 0, 1];
const initialPosition = [30, 21, 5];
const state = () => page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__?.state());
const close = (actual, expected, tolerance = 1e-4) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, axis) => assert(Math.abs(value - expected[axis]) < tolerance, `${actual} != ${expected}`));
};
const unit = vector => vector.map(value => value / Math.hypot(...vector));
const direction = (from, to) => unit(to.map((value, axis) => value - from[axis]));
const screenTransform = { 'flip-horizontal': ([x, y]) => [-x, y], 'flip-vertical': ([x, y]) => [x, -y],
  'rotate-clockwise': ([x, y]) => [-y, x], 'rotate-counterclockwise': ([x, y]) => [y, -x] };
const nextMatrix = { 'flip-horizontal': ([a, b, c, d]) => [-a, -b, c, d],
  'flip-vertical': ([a, b, c, d]) => [a, b, -c, -d],
  'rotate-clockwise': ([a, b, c, d]) => [c, d, -a, -b],
  'rotate-counterclockwise': ([a, b, c, d]) => [-c, -d, a, b],
  'reset-orientation': () => identity };
const nextLabels = { 'flip-horizontal': ({ top, bottom, left, right }) => ({ top, bottom, left: right, right: left }),
  'flip-vertical': ({ top, bottom, left, right }) => ({ top: bottom, bottom: top, left, right }),
  'rotate-clockwise': ({ top, bottom, left, right }) => ({ top: left, bottom: right, left: bottom, right: top }),
  'rotate-counterclockwise': ({ top, bottom, left, right }) => ({ top: right, bottom: left, left: top, right: bottom }) };
let study, prefix;
const get = async route => { const response = await page.request.get(prefix + route); assert(response.ok(), await response.text()); return response; };
const metadata = async () => (await get('')).json();
const masks = async time => (await get('/masks?time=' + time)).body();
async function check(name, work) { await work(); report.checks.push(name); console.log('PASS', name); }
async function ready() { await expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 90000 }); }
async function idle() { await expect(page.getByTestId('segment-all')).toBeEnabled({ timeout: 30000 }); }
async function value(label, next) {
  await page.getByLabel(label, { exact: true }).evaluate((element, input) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(element, String(input));
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }, next);
}
async function jump(time) {
  await value('Time frame', time);
  await expect.poll(async () => (await state()).time, { timeout: 30000 }).toBe(time);
  await expect(page.getByRole('button', { name: 'Play cine', exact: true })).toBeEnabled({ timeout: 30000 });
}
async function coordinates(view, indices) {
  return page.evaluate(({ view, indices }) => indices.map(index =>
    window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas(view, index)), { view, indices });
}
async function clickIndex(view, index) {
  const [point] = await coordinates(view, [index]);
  const box = await page.getByTestId('viewport-' + view).boundingBox();
  await page.mouse.click(box.x + point[0], box.y + point[1]);
}
async function fit(view) {
  await page.getByLabel('Fit ' + views[view].title, { exact: true }).click();
  // Expanding a pane schedules the rendering engine's resize on the next tick.
  await page.waitForTimeout(160);
}
async function action(view, operation) {
  const before = (await state()).displayOrientations[view];
  await page.getByTestId('orientation-' + view).click();
  await page.getByTestId(operation + '-' + view).click();
  await expect.poll(async () => (await state()).displayOrientations[view].map(value => value || 0))
    .toEqual(nextMatrix[operation](before).map(value => value || 0));
  await page.getByTestId('orientation-' + view).click();
}
async function samples(view, indices, filename) {
  const points = await coordinates(view, indices);
  const png = await page.getByTestId('viewport-' + view).screenshot({
    path: path.join(output, filename), timeout: 30000, style: '.viewport-overlay { visibility: hidden !important; }',
  });
  return page.evaluate(async ({ encoded, points }) => {
    const image = new Image(); image.src = 'data:image/png;base64,' + encoded; await image.decode();
    const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext('2d', { willReadFrequently: true }); context.drawImage(image, 0, 0);
    return points.map(point => {
      const [x, y] = point.map(Math.round);
      if (x < 1 || y < 1 || x >= canvas.width - 1 || y >= canvas.height - 1) return { point, outside: true };
      const bytes = context.getImageData(x - 1, y - 1, 3, 3).data;
      return { point, rgb: [0, 1, 2].map(channel => Array.from({ length: 9 }, (_, n) => bytes[n * 4 + channel])
        .sort((a, b) => a - b)[4]) };
    });
  }, { encoded: png.toString('base64'), points });
}
async function assertPixelSamples(view, stage) {
  const current = await state(), { axis } = views[view];
  const indices = [[13, 13, 1], [45, 37, 10], [30, 21, 5]].map(index => {
    index[axis] = current.position[axis]; return index;
  });
  const observed = await samples(view, indices, `${view}-${stage}.png`);
  observed.forEach(sample => assert(!sample.outside, `${view}/${stage}: voxel outside fitted viewport`));
  for (let i = 0; i < 2; i++) {
    const [x, y, z] = indices[i], intensity = 15 + Math.floor(x / 8) * 16 + Math.floor(y / 8) * 35 + Math.floor(z / 3) * 65 + current.time * 20;
    const gray = Math.max(0, Math.min(255, (intensity - current.window.lower) / (current.window.upper - current.window.lower) * 255));
    observed[i].rgb.forEach(value => assert(Math.abs(value - gray) < 10,
      `${view}/${stage}: voxel ${indices[i]} expected grayscale ${gray}, saw ${observed[i].rgb}`));
  }
  const color = observed[2].rgb;
  assert(Math.max(...color) - Math.min(...color) > 20, `${view}/${stage}: MYO overlay missing at known label voxel: ${color}`);
  report.pixels.push({ view, stage, time: current.time, indices, observed });
  return observed;
}
function assertDirections(actual, expected) {
  for (const name of Object.keys(views)) assert.deepEqual(actual.displayOrientations[name].map(value => value || 0),
    expected.displayOrientations[name].map(value => value || 0));
}

try {
  const uploaded = await page.request.post(base + '/api/projects', { multipart: {
    file: { name: 'orientation-controls.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(fixture.path) },
  } });
  assert(uploaded.ok(), await uploaded.text()); study = await uploaded.json(); prefix = base + '/api/projects/' + study.id;
  const imported = await page.request.post(prefix + '/annotations', { multipart: { revision: String(study.revision), target: 'annotation', binary_organ: '0',
    file: { name: 'orientation-labels.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(labelsPath) },
  } });
  assert(imported.ok(), await imported.text()); study = await imported.json(); report.study_id = study.id;
  await page.addInitScript(({ id, position }) => {
    const key = 'kprism-view-' + id;
    if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify({ tool: 'crosshair', position }));
  }, { id: study.id, position: initialPosition });
  await page.goto(base + '/?study=' + study.id + '&debug=1'); await ready();
  const originalMasks = await Promise.all([masks(0), masks(1)]);
  const original = await state();
  for (const view of Object.keys(views)) assert.deepEqual(original.displayOrientations[view], identity);

  for (const [view, { axis, title }] of Object.entries(views)) {
    await check(`${title}: horizontal / vertical flips and quarter turns move image, mask and patient labels together`, async () => {
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      await fit(view);
      const baselinePixels = await assertPixelSamples(view, 'original');
      for (const operation of ['flip-horizontal', 'flip-vertical', 'rotate-clockwise', 'rotate-counterclockwise']) {
        const before = await state();
        const indices = [[13, 13, 1], [45, 37, 10]].map(index => { index[axis] = before.position[axis]; return index; });
        const [first, second] = await coordinates(view, indices);
        const camera = before.sliceCameras.find(item => item.name === view).camera;
        await action(view, operation);
        const after = await state(), nextCamera = after.sliceCameras.find(item => item.name === view).camera;
        close(after.position, before.position); close(after.locator.center, before.locator.center);
        assert.equal(nextCamera.parallelScale, camera.parallelScale);
        close(nextCamera.focalPoint, camera.focalPoint);
        for (const other of Object.keys(views).filter(name => name !== view)) {
          assert.deepEqual(after.displayOrientations[other], before.displayOrientations[other]);
          const a = after.sliceCameras.find(item => item.name === other).camera, b = before.sliceCameras.find(item => item.name === other).camera;
          close(a.viewUp, b.viewUp); close(a.viewPlaneNormal, b.viewPlaneNormal);
        }
        const [movedFirst, movedSecond] = await coordinates(view, indices);
        close(direction(movedFirst, movedSecond), screenTransform[operation](direction(first, second)), 1e-3);
        assert.deepEqual(after.patientDirections[view], nextLabels[operation](before.patientDirections[view]));
        await fit(view);
        const transformedPixels = await assertPixelSamples(view, operation);
        transformedPixels.forEach((sample, index) => close(sample.rgb, baselinePixels[index].rgb, 10));
      }
      await action(view, 'reset-orientation');
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      close((await state()).position, initialPosition);
    });
  }
  await check('Display changes leave source geometry, masks and annotation revision unchanged', async () => {
    assert.equal((await metadata()).revision, study.revision);
    assert.deepEqual((await metadata()).source_geometry, study.source_geometry);
    for (const time of [0, 1]) assert.deepEqual(await masks(time), originalMasks[time]);
    assert.equal((await state()).sourceUploads, original.sourceUploads);
    assert.equal((await state()).surfaceCache.builds, original.surfaceCache.builds);
  });

  await action('native', 'flip-horizontal'); await action('native', 'rotate-clockwise');
  await action('coronal', 'flip-vertical'); await action('sagittal', 'rotate-counterclockwise');
  await check('Flips and rotations preserve original acquisition-order wheel browsing on all three planes', async () => {
    for (const [view, { axis, title }] of Object.entries(views)) {
      await page.getByLabel('Expand ' + title, { exact: true }).click(); await fit(view);
      const before = await state();
      const normal = original.sliceCameras.find(item => item.name === view).camera.viewPlaneNormal;
      const basis = before.geometry.direction.slice(axis * 3, axis * 3 + 3);
      const step = view === 'native' ? 1 : Math.sign(normal.reduce((sum, value, index) => sum + value * basis[index], 0));
      const box = await page.getByTestId('viewport-' + view).boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.wheel(0, 120);
      await expect.poll(async () => (await state()).position[axis]).toBeCloseTo(before.position[axis] + step, 3);
      const expected = [...before.position]; expected[axis] += step; close((await state()).position, expected);
      await page.mouse.wheel(0, -120);
      await expect.poll(async () => (await state()).position[axis]).toBeCloseTo(before.position[axis], 3);
      assertDirections(await state(), before);
      await page.getByLabel('Expand ' + title, { exact: true }).click();
    }
  });
  await check('Pan follows screen drag and zoom retains the transformed image basis without moving the shared cursor', async () => {
    await page.getByLabel('Expand Acquisition plane', { exact: true }).click(); await fit('native');
    const before = await state(), indices = [[20, 15, before.position[2]], [40, 30, before.position[2]]];
    const initial = await coordinates('native', indices);
    await page.getByLabel('Pan (6)', { exact: true }).click();
    const box = await page.getByTestId('viewport-native').boundingBox();
    const start = [box.x + box.width / 2, box.y + box.height / 2];
    await page.mouse.move(...start); await page.mouse.down();
    await page.mouse.move(start[0] + 40, start[1] + 25, { steps: 8 }); await page.mouse.up();
    const panned = await coordinates('native', indices);
    close(panned[0].map((value, axis) => value - initial[0][axis]), [40, 25], 2);
    close(direction(...panned), direction(...initial), 1e-4);
    close((await state()).position, before.position); assertDirections(await state(), before);
    await page.getByLabel('Zoom (7)', { exact: true }).click();
    const scale = (await state()).sliceCameras.find(item => item.name === 'native').camera.parallelScale;
    await page.mouse.move(...start); await page.mouse.down();
    await page.mouse.move(start[0], start[1] + 50, { steps: 8 }); await page.mouse.up();
    await expect.poll(async () => (await state()).sliceCameras.find(item => item.name === 'native').camera.parallelScale).not.toBe(scale);
    const zoomed = await coordinates('native', indices);
    close(direction(...zoomed), direction(...initial), 1e-4);
    close((await state()).position, before.position); assertDirections(await state(), before);
    await fit('native'); await page.getByLabel('Locate (3)', { exact: true }).click();
    await page.getByLabel('Expand Acquisition plane', { exact: true }).click();
  });
  await check('Orientation controls remain visible and their menus scroll on laptop-sized layouts', async () => {
    for (const width of [1366, 1050]) {
      await page.setViewportSize({ width, height: 768 }); await page.waitForTimeout(160);
      for (const view of Object.keys(views)) {
        const button = page.getByTestId('orientation-' + view);
        await expect(button).toBeInViewport({ ratio: 1 });
        const box = await button.boundingBox(), pane = await page.getByTestId('pane-' + view).boundingBox();
        assert(box.x >= pane.x && box.x + box.width <= pane.x + pane.width + 1, `${view} orientation button extends beyond pane at ${width}px`);
        await button.click();
        const panel = page.locator('#orientation-panel-' + view);
        const geometry = await panel.evaluate(element => ({ client: element.clientHeight, scroll: element.scrollHeight,
          overflow: getComputedStyle(element).overflowY, box: element.getBoundingClientRect().toJSON() }));
        if (geometry.scroll > geometry.client + 1) assert(['auto', 'scroll'].includes(geometry.overflow));
        await page.getByTestId('reset-orientation-' + view).scrollIntoViewIfNeeded();
        await expect(page.getByTestId('reset-orientation-' + view)).toBeInViewport({ ratio: 1 });
        assert(geometry.box.x >= 0 && geometry.box.right <= width + 1);
        await button.click();
      }
      await saveDiagnosticScreenshot(page, { path: path.join(output, `orientation-layout-${width}.png`) }, report);
    }
    await page.setViewportSize({ width: 1680, height: 1060 }); await page.waitForTimeout(160);
  });
  await check('Independent directions survive fit, expansion, resize, cine and reload', async () => {
    const expected = await state();
    for (const [view, { title }] of Object.entries(views)) {
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      await fit(view);
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      assertDirections(await state(), expected);
    }
    await page.setViewportSize({ width: 1500, height: 960 }); await page.waitForTimeout(160);
    assertDirections(await state(), expected); close((await state()).position, expected.position);
    await page.setViewportSize({ width: 1680, height: 1060 });
    await jump(1); assertDirections(await state(), expected);
    const time = (await state()).time;
    await page.getByRole('button', { name: 'Play cine', exact: true }).click();
    await expect.poll(async () => (await state()).time, { timeout: 30000, intervals: [30] }).not.toBe(time);
    await page.getByRole('button', { name: 'Pause cine', exact: true }).click();
    assertDirections(await state(), expected); await jump(0);
    await page.reload(); await ready(); assertDirections(await state(), expected);
    const persisted = await page.evaluate(id => JSON.parse(localStorage.getItem('cardiac-prism-orientation-' + id)), study.id);
    assert.equal(persisted.version, 1); assert.deepEqual(persisted.views, expected.displayOrientations);
    for (const [view, { title }] of Object.entries(views)) {
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      await fit(view);
      await assertPixelSamples(view, 'restored');
      await page.getByLabel('Expand ' + title, { exact: true }).click();
    }
  });
  await check('Clicks in each transformed view move the shared cursor to the original acquisition voxel', async () => {
    await page.getByLabel('Locate (3)', { exact: true }).click();
    for (const [view, { axis, title }] of Object.entries(views)) {
      await page.getByLabel('Expand ' + title, { exact: true }).click();
      await fit(view);
      const target = [35, 25, 6]; target[axis] = (await state()).position[axis];
      await clickIndex(view, target);
      await expect.poll(async () => Math.max(...(await state()).position.map((value, index) => Math.abs(value - target[index]))))
        .toBeLessThan(.01);
      const current = await state();
      const world = await page.evaluate(index => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToWorld(index), current.position);
      close(current.locator.center, world);
      const [point] = await coordinates(view, [current.position]);
      const circle = page.getByTestId('viewport-' + view).getByTestId('view-locator').locator('circle').first();
      close([Number(await circle.getAttribute('cx')), Number(await circle.getAttribute('cy'))], point);
      await page.getByLabel('Expand ' + title, { exact: true }).click();
    }
  });
  await page.getByLabel('Expand Acquisition plane', { exact: true }).click();
  await fit('native');
  await page.getByLabel('Select MYO', { exact: true }).click();
  await check('Positive and negative prompt requests retain original coordinates under a rotated reflection', async () => {
    // Only the inference transport is stubbed: actual mouse picking and request
    // serialization are exercised without loading a model in browser CI.
    await page.route('**/api/projects/' + study.id + '/inference', async route => {
      const request = route.request().postDataJSON(); report.prompts.push(request);
      await route.fulfill({ json: { id: 'orientation-coordinate-test', state: 'queued', seconds: 0 } });
    });
    await page.route('**/api/jobs/orientation-coordinate-test', route => route.fulfill({
      json: { id: 'orientation-coordinate-test', state: 'completed', seconds: 0, changed_voxels: 0 },
    }));
    for (const positive of [true, false]) {
      const target = [16, 14, (await state()).position[2]];
      await page.getByLabel(positive ? 'Include (1)' : 'Exclude (2)', { exact: true }).click();
      await clickIndex('native', target);
      await expect.poll(() => report.prompts.length).toBe(positive ? 1 : 2); await idle();
      const request = report.prompts.at(-1);
      close([request.point.x, request.point.y], target.slice(0, 2), .01);
      assert.equal(request.point.positive, positive); assert.equal(request.time, 0); assert.equal(request.slice, target[2]);
      assert.equal(request.organ, 2);
    }
    await page.unroute('**/api/projects/' + study.id + '/inference');
    await page.unroute('**/api/jobs/orientation-coordinate-test');
  });
  let brushedVoxel;
  await check('One-pixel brush, erase and undo edit only the visible original voxel in the selected time frame', async () => {
    await page.getByLabel('Brush (4)', { exact: true }).click();
    const before = await masks(0), revision = (await metadata()).revision;
    const target = [16, 14, (await state()).position[2]], [frames, depth, height, width] = study.shape;
    const offset = depth * height * width + (target[2] * height + target[1]) * width + target[0];
    assert.equal(before[offset], 0);
    await clickIndex('native', target);
    await expect.poll(async () => (await metadata()).revision).toBe(revision + 1); await idle();
    const expected = Buffer.from(before); expected[offset] = 1;
    assert.deepEqual(await masks(0), expected); assert.deepEqual(await masks(frames - 1), originalMasks[1]);
    await page.getByLabel('Erase (5)', { exact: true }).click(); await clickIndex('native', target);
    await expect.poll(async () => (await metadata()).revision).toBe(revision + 2); await idle();
    assert.deepEqual(await masks(0), before);
    await page.getByLabel('Undo edit', { exact: true }).click(); await idle();
    assert.deepEqual(await masks(0), expected); brushedVoxel = [0, target[2], target[1], target[0]];
  });
  await check('Export keeps original 4D geometry and exactly the original labels plus the edited voxel', async () => {
    const exported = path.join(output, 'transformed-view-export.nii.gz');
    fs.writeFileSync(exported, await (await get('/export?kind=labelmap')).body());
    const result = execFileSync(python, ['-c',
      'import sys,json,SimpleITK as s,numpy as n;i=s.ReadImage(sys.argv[1]);e=s.ReadImage(sys.argv[2]);a=s.GetArrayFromImage(i);b=s.GetArrayFromImage(e);v=json.loads(sys.argv[3]);a[tuple(v)]=2;assert n.array_equal(a,b);assert i.GetSize()==e.GetSize();assert n.allclose(i.GetSpacing(),e.GetSpacing(),rtol=0,atol=1e-5);assert n.allclose(i.GetOrigin(),e.GetOrigin(),rtol=0,atol=1e-5);assert n.allclose(i.GetDirection(),e.GetDirection(),rtol=0,atol=1e-5);print(json.dumps({"original_geometry":True,"labels_exact":True,"edited_voxel_tzyx":v}))',
      labelsPath, exported, JSON.stringify(brushedVoxel)], { encoding: 'utf8' });
    report.export = JSON.parse(result);
  });
  await check('Reset affects only the chosen view and persists its original orientation', async () => {
    const before = await state(); await action('native', 'reset-orientation');
    const after = await state();
    assert.deepEqual(after.displayOrientations.native, identity);
    for (const view of ['coronal', 'sagittal']) assert.deepEqual(after.displayOrientations[view], before.displayOrientations[view]);
    assert.deepEqual(after.patientDirections.native, original.patientDirections.native);
    await page.reload(); await ready(); assertDirections(await state(), after);
  });
  await saveDiagnosticScreenshot(page, { path: path.join(output, 'orientation-controls.png') }, report);
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) {
  report.failure = error.stack; process.exitCode = 1;
  await saveDiagnosticScreenshot(page, { path: path.join(output, 'orientation-controls-failure.png') }, report).catch(() => {});
} finally {
  fs.writeFileSync(path.join(output, 'orientation-controls.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors, failure: report.failure }));
  await browser.close();
}
