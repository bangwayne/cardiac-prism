// Display orientation must not alter the original voxel addressed by any edit.
// Each transformed operation is compared with the identical operation in the
// original view, including all three labels and untouched slices/time frames.
import { chromium, expect } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { base, fixturePath, reportDirectory, browserOptions, assertIsolatedService } from './browser_helpers.mjs';
import { saveDiagnosticScreenshot } from './browser_artifacts.mjs';

const output = fileURLToPath(reportDirectory('orientation-edit-invariance'));
const python = process.env.CARDIAC_PRISM_TEST_PYTHON || 'python3';
const fixtures = JSON.parse(fs.readFileSync(fixturePath('orientation', 'manifest.json'), 'utf8'));
const aligned = path.join(output, 'aligned-3d.nii.gz');
execFileSync(python, ['-c', 'import sys,SimpleITK as s;i=s.ReadImage(sys.argv[1]);o=i[:,:,:,0];s.WriteImage(o,sys.argv[2])',
  fixtures.find(item => item.name === 'identity-aligned').path, aligned]);
const geometries = [...fixtures.filter(item => ['oblique-reflected', 'oblique-positive'].includes(item.name)),
  { name: 'aligned-3d', path: aligned }];
const transforms = Array.from({ length: 8 }, (_, index) => ({
  name: `${index < 4 ? 'rotation' : 'reflection'}-${index % 4 * 90}`,
  operations: [...(index >= 4 ? ['flip-horizontal'] : []), ...Array(index % 4).fill('rotate-clockwise')],
}));
const identity = [1, 0, 0, 1];
const nextMatrix = (matrix, operation) => {
  const [a, b, c, d] = matrix;
  return (operation === 'flip-horizontal' ? [-a, -b, c, d] : [c, d, -a, -b]).map(value => value || 0);
};
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { passed: false, isolated: true, checks: [], geometries: [], errors: [] };
await assertIsolatedService();
const browser = await chromium.launch(browserOptions);
let page;
try {
  for (const fixture of geometries) {
    page = await browser.newPage({ viewport: { width: 1680, height: 1060 } });
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    page.on('crash', () => report.errors.push('Browser page crashed'));
    const record = { name: fixture.name, states: [], prompts: [], exports: [] };
    report.geometries.push(record);
    const uploaded = await page.request.post(base + '/api/projects', { multipart: {
      file: { name: `${fixture.name}-edit-invariance.nii.gz`, mimeType: 'application/octet-stream', buffer: fs.readFileSync(fixture.path) },
    } });
    assert(uploaded.ok(), await uploaded.text());
    let study = await uploaded.json();
    const prefix = base + '/api/projects/' + study.id;
    const labelsPath = path.join(output, fixture.name + '-labels.nii.gz');
    execFileSync(python, ['-c',
      'import sys,SimpleITK as s,numpy as n;i=s.ReadImage(sys.argv[1]);a=n.zeros(s.GetArrayFromImage(i).shape,n.uint8);a[...,3:8,28:32,32:36]=3;a[...,2:5,8:12,9:13]=1;a[...,8:10,35:39,43:47]=2;o=s.GetImageFromArray(a,isVector=False);o.CopyInformation(i);s.WriteImage(o,sys.argv[2])',
      fixture.path, labelsPath]);
    const imported = await page.request.post(prefix + '/annotations', { multipart: { revision: String(study.revision), target: 'annotation', binary_organ: '0',
      file: { name: fixture.name + '-labels.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(labelsPath) },
    } });
    assert(imported.ok(), await imported.text()); study = await imported.json();
    const [frames, depth, height, width] = study.shape, volumeSize = depth * height * width;
    const time = frames - 1, layer = 6;
    const get = async route => { const response = await page.request.get(prefix + route); assert(response.ok(), await response.text()); return response; };
    const metadata = async () => (await get('')).json();
    const masks = async t => (await get('/masks?time=' + t)).body();
    const allMasks = async () => Promise.all(Array.from({ length: frames }, (_, t) => masks(t)));
    const sourceHashes = async () => Promise.all(Array.from({ length: frames }, async (_, t) => digest(await (await get('/volume?time=' + t)).body())));
    const state = () => page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__?.state());
    const ready = async () => expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 90000 });
    const idle = async () => expect(page.getByTestId('segment-all')).toBeEnabled({ timeout: 30000 });
    const originalMasks = await allMasks(), originalSourceHashes = await sourceHashes();
    await page.addInitScript(({ id, time, layer }) => {
      const key = 'kprism-view-' + id;
      if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify({ tool: 'crosshair', time, position: [30, 24, layer], brushRadius: 0 }));
    }, { id: study.id, time, layer });
    await page.goto(base + '/?study=' + study.id + '&debug=1'); await ready();
    await page.getByLabel('Expand Acquisition plane', { exact: true }).click();
    await page.getByLabel('Fit Acquisition plane', { exact: true }).click();
    await page.waitForTimeout(160);
    const initialState = await state();
    assert.equal(initialState.time, time); assert.equal(initialState.position[2], layer);
    const setInput = async (label, value) => page.getByLabel(label, { exact: true }).evaluate((element, input) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(element, String(input));
      element.dispatchEvent(new Event('input', { bubbles: true }));
    }, value);
    async function orient(transform) {
      await page.getByTestId('orientation-native').click();
      await page.getByTestId('reset-orientation-native').click();
      for (const operation of transform.operations) await page.getByTestId(operation + '-native').click();
      await page.getByTestId('orientation-native').click();
      const expected = transform.operations.reduce(nextMatrix, identity);
      await expect.poll(async () => (await state()).displayOrientations.native).toEqual(expected);
      return expected;
    }
    async function screen(indices) {
      const points = await page.evaluate(indices => indices.map(index =>
        window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas('native', index)), indices);
      const box = await page.getByTestId('viewport-native').boundingBox();
      return points.map(point => {
        assert(point.every(Number.isFinite), 'Non-finite transformed canvas point');
        assert(point[0] > 0 && point[0] < box.width && point[1] > 0 && point[1] < box.height, 'Edit target outside native viewport');
        return [box.x + point[0], box.y + point[1]];
      });
    }
    async function changed(work) {
      const revision = (await metadata()).revision;
      await work();
      await expect.poll(async () => (await metadata()).revision, { timeout: 30000 }).toBe(revision + 1);
      await idle();
    }
    async function stroke(organ, radius, erase, points) {
      await page.getByLabel('Select ' + ['RV', 'MYO', 'LV'][organ - 1], { exact: true }).click();
      await page.getByLabel(erase ? 'Erase (5)' : 'Brush (4)', { exact: true }).click();
      await setInput(erase ? 'Eraser size' : 'Brush size', radius);
      const positions = await screen(points.map(([x, y]) => [x, y, layer]));
      await changed(async () => {
        await page.mouse.move(...positions[0]); await page.mouse.down();
        for (const point of positions.slice(1)) await page.mouse.move(...point);
        await page.mouse.up();
      });
    }
    async function history(redo = false) {
      await changed(() => page.getByLabel(redo ? 'Redo edit' : 'Undo edit', { exact: true }).click());
    }
    async function assertMasks(expected, label) {
      const actual = await allMasks();
      for (let t = 0; t < frames; t++) assert.deepEqual(actual[t], expected[t], `${fixture.name}/${label}: frame ${t} original mask bytes`);
    }
    function assertTargetScope(actual, label) {
      for (let t = 0; t < frames; t++) for (let organ = 0; organ < 3; organ++) for (let z = 0; z < depth; z++) {
        if (t === time && z === layer) continue;
        const start = organ * volumeSize + z * height * width;
        assert.deepEqual(actual[t].subarray(start, start + height * width),
          originalMasks[t].subarray(start, start + height * width), `${label}: changed another frame or layer`);
      }
    }
    const onePixelPath = [[20, 16], [21, 16], [22, 16], [22, 17]];
    const finitePath = [[29, 27], [32, 27], [35, 30], [38, 30]];
    const erasePath = [[32, 27], [32, 30], [35, 30]];
    let identityPaint, identityErase, finalMasks;
    const inferencePattern = '**/api/projects/' + study.id + '/inference';
    await page.route(inferencePattern, async route => {
      record.prompts.push(route.request().postDataJSON());
      await route.fulfill({ json: { id: 'orientation-edit-prompts', state: 'queued', seconds: 0 } });
    });
    await page.route('**/api/jobs/orientation-edit-prompts', route => route.fulfill({
      json: { id: 'orientation-edit-prompts', state: 'completed', seconds: 0, changed_voxels: 0 },
    }));
    for (let index = 0; index < transforms.length; index++) {
      const transform = transforms[index], beforeOrientation = await metadata();
      const matrix = await orient(transform);
      const afterOrientation = await metadata();
      assert.equal(afterOrientation.revision, beforeOrientation.revision);
      assert.deepEqual(afterOrientation.source_geometry, study.source_geometry);
      assert.deepEqual(await sourceHashes(), originalSourceHashes);
      await assertMasks(originalMasks, transform.name + '/orientation-only');
      // Same original voxel, different screen location. Only model execution is
      // stubbed; mouse picking, polarity, selected label and HTTP serialization run.
      for (const positive of [true, false]) {
        const organ = index % 3 + 1, count = record.prompts.length;
        await page.getByLabel('Select ' + ['RV', 'MYO', 'LV'][organ - 1], { exact: true }).click();
        await page.getByLabel(positive ? 'Include (1)' : 'Exclude (2)', { exact: true }).click();
        const [point] = await screen([[17, 13, layer]]);
        await page.mouse.click(...point);
        await expect.poll(() => record.prompts.length).toBe(count + 1); await idle();
        const prompt = record.prompts.at(-1);
        assert(Math.abs(prompt.point.x - 17) < .01 && Math.abs(prompt.point.y - 13) < .01);
        assert.equal(prompt.point.positive, positive); assert.equal(prompt.organ, organ);
        assert.equal(prompt.time, time); assert.equal(prompt.slice, layer);
      }
      // Explicitly expected original coordinates (not derived from camera APIs).
      for (const organ of [1, 2, 3]) {
        const expected = originalMasks.map(bytes => Buffer.from(bytes));
        for (const [x, y] of onePixelPath) expected[time][(organ - 1) * volumeSize + (layer * height + y) * width + x] = 1;
        await stroke(organ, 0, false, onePixelPath);
        await assertMasks(expected, transform.name + `/one-pixel-brush-${organ}`);
        if (organ === 3) {
          await history(); await assertMasks(originalMasks, transform.name + '/undo-brush');
          await history(true); await assertMasks(expected, transform.name + '/redo-brush');
        }
        await stroke(organ, 0, true, onePixelPath);
        await assertMasks(originalMasks, transform.name + `/one-pixel-erase-${organ}`);
        if (organ === 3) {
          await history(); await assertMasks(expected, transform.name + '/undo-erase');
          await history(true); await assertMasks(originalMasks, transform.name + '/redo-erase');
        }
      }
      await stroke(2, 3, false, finitePath);
      const painted = await allMasks();
      if (index === 0) {
        identityPaint = painted; assertTargetScope(identityPaint, 'Finite brush');
        assert(painted[time].subarray(volumeSize, 2 * volumeSize).some(value => value), 'Finite brush must add MYO');
        assert(painted[time].subarray(2 * volumeSize).reduce((sum, value) => sum + value, 0)
          < originalMasks[time].subarray(2 * volumeSize).reduce((sum, value) => sum + value, 0), 'MYO brush must overwrite pre-existing LV');
      }
      await assertMasks(identityPaint, transform.name + '/finite-brush-and-overwrite');
      await history(); await assertMasks(originalMasks, transform.name + '/undo-overwrite');
      await history(true); await assertMasks(identityPaint, transform.name + '/redo-overwrite');
      await stroke(2, 2, true, erasePath);
      if (index === 0) {
        identityErase = await allMasks(); assertTargetScope(identityErase, 'Finite eraser');
        assert(!identityErase[time].equals(identityPaint[time]), 'Finite-radius eraser must change the mask');
      }
      await assertMasks(identityErase, transform.name + '/finite-erase');
      await history(); await assertMasks(identityPaint, transform.name + '/undo-finite-erase');
      await history(true); await assertMasks(identityErase, transform.name + '/redo-finite-erase');
      if (index < transforms.length - 1) {
        await history(); await history(); await assertMasks(originalMasks, transform.name + '/restored');
      } else finalMasks = await allMasks();
      record.states.push({ name: transform.name, matrix, nativeVoxelEdits: true, exactIdentityResult: true, frame: time, layer });
      console.log(`PASS ${fixture.name}: ${transform.name}; exact three-label brush/erase, drag, overwrite, undo/redo and prompt coordinates`);
    }
    assert.equal(new Set(record.states.map(item => item.matrix.join(','))).size, 8);
    assert.deepEqual(await sourceHashes(), originalSourceHashes);
    assert.deepEqual((await metadata()).source_geometry, study.source_geometry);
    await page.getByRole('button', { name: 'Save study', exact: true }).click(); await idle();
    const savedOrientation = (await state()).displayOrientations;
    await page.reload(); await ready();
    assert.deepEqual((await state()).displayOrientations, savedOrientation);
    await assertMasks(finalMasks, 'save-reload');
    assert.deepEqual(await sourceHashes(), originalSourceHashes);
    const exported = path.join(output, fixture.name + '-export.nii.gz');
    const expectedRaw = path.join(output, fixture.name + '-expected-masks.bin');
    fs.writeFileSync(exported, await (await get('/export?kind=labelmap')).body());
    fs.writeFileSync(expectedRaw, Buffer.concat(finalMasks));
    const result = execFileSync(python, ['-c',
      'import sys,json,SimpleITK as s,numpy as n;i=s.ReadImage(sys.argv[1]);e=s.ReadImage(sys.argv[2]);shape=json.loads(sys.argv[4]);t,z,y,x=shape;raw=n.fromfile(sys.argv[3],n.uint8).reshape(t,3,z,y,x);a=n.zeros((t,z,y,x),n.uint8);[(n.copyto(a,n.uint8(k+1),where=raw[:,k].astype(bool))) for k in range(3)];a=a[0] if i.GetDimension()==3 else a;assert n.array_equal(a,s.GetArrayFromImage(e));assert i.GetSize()==e.GetSize();assert n.allclose(i.GetSpacing(),e.GetSpacing(),rtol=0,atol=1e-5);assert n.allclose(i.GetOrigin(),e.GetOrigin(),rtol=0,atol=1e-5);assert n.allclose(i.GetDirection(),e.GetDirection(),rtol=0,atol=1e-5);print(json.dumps({"original_geometry":True,"all_labels_exact":True,"size":e.GetSize()}))',
      fixture.path, exported, expectedRaw, JSON.stringify(study.shape)], { encoding: 'utf8' });
    record.exports.push(JSON.parse(result)); record.originalSourceHashes = originalSourceHashes;
    report.checks.push(`${fixture.name}: all eight orientations preserve source data and exact original-space edits, history, save/reload and export`);
    console.log('PASS', report.checks.at(-1));
    await page.close(); page = null;
  }
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) {
  report.failure = error.stack; process.exitCode = 1;
  if (page) await saveDiagnosticScreenshot(page, { path: path.join(output, 'edit-invariance-failure.png') }, report).catch(() => {});
} finally {
  fs.writeFileSync(path.join(output, 'edit-invariance.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors, failure: report.failure }));
  await browser.close();
}
