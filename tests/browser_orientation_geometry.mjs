// Physical-coordinate, interaction and export checks on an asymmetric synthetic cine.
import { chromium, expect } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import { base, fixturePath, reportDirectory, browserOptions, assertIsolatedService } from './browser_helpers.mjs';
import { saveDiagnosticScreenshot } from './browser_artifacts.mjs';
import { fileURLToPath } from 'node:url';
const output = fileURLToPath(reportDirectory('orientation-geometry'));
const privateDir = output;
await assertIsolatedService();
const fixtures = JSON.parse(fs.readFileSync(fixturePath('orientation', 'manifest.json'), 'utf8'));
const fixture = fixtures.find(item => item.name === 'oblique-reflected');
const form = new FormData();
form.append('file', new Blob([fs.readFileSync(fixture.path)]), 'oblique-cine.nii.gz');
const uploaded = await fetch(base + '/api/projects', { method: 'POST', body: form });
assert(uploaded.ok, await uploaded.clone().text());
const study = await uploaded.json();
const prefix = base + '/api/projects/' + study.id;
const report = { passed: false, checks: [], errors: [], study_id: study.id, isolated: true };
const browser = await chromium.launch(browserOptions);
const page = await browser.newPage({ viewport: { width: 1920, height: 1400 } });
page.on('pageerror', error => report.errors.push(error.stack || error.message));
page.on('crash', () => report.errors.push('Browser page crashed'));
const state = () => page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__?.state());
const get = async route => { const response = await page.request.get(prefix + route); assert(response.ok(), await response.text()); return response; };
const metadata = async () => (await get('')).json();
const masks = async t => new Uint8Array(await (await get('/masks?time=' + t)).body());
const dot = (a, b) => a.reduce((sum, n, i) => sum + n * b[i], 0);
const close = (actual, expected, tolerance = 1e-4) => actual.forEach((n, i) => assert(Math.abs(n - expected[i]) < tolerance, `${actual} != ${expected}`));
const [frames, depth, height, width] = study.shape, volume = depth * height * width;
function toWorld(index) {
  const geometry = study.source_geometry, n = geometry.dimension;
  return geometry.origin.slice(0, 3).map((value, row) => value + index.reduce((sum, value, col) =>
    sum + value * geometry.spacing[col] * geometry.direction[row * n + col], 0));
}
async function check(name, work) { await work(); report.checks.push(name); console.log('PASS', name); }
async function ready() { await expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 120000 }); }
async function idle() { await expect(page.getByTestId('segment-all')).toBeEnabled({ timeout: 30000 }); }
async function clickIndex(name, index) {
  const canvas = await page.evaluate(({ name, index }) => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas(name, index), { name, index });
  const box = await page.getByTestId('viewport-' + name).boundingBox();
  await page.mouse.click(box.x + canvas[0], box.y + canvas[1]);
}
try {
  await assertIsolatedService();
  assert.deepEqual((await metadata()).source_geometry, study.source_geometry);
  await page.goto(base + '/?study=' + study.id + '&debug=1'); await ready();
  await check('Oblique reflected native / X / Y views use mutually orthogonal acquisition axes in exact physical coordinates', async () => {
    const d = await state(); assert.equal(d.reconstructionGeometry.convention, 'acquisition');
    assert.equal(d.reconstructionGeometry.patientAxesAligned, false);
    const axes = [0, 1, 2].map(axis => d.geometry.direction.slice(axis * 3, axis * 3 + 3));
    const cameras = d.sliceCameras.map(view => view.camera);
    [2, 1, 0].forEach((axis, view) => assert(Math.abs(Math.abs(dot(cameras[view].viewPlaneNormal, axes[axis])) - 1) < 1e-6));
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) assert(Math.abs(dot(cameras[i].viewPlaneNormal, cameras[j].viewPlaneNormal)) < 1e-6);
    const observed = await page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToWorld([30, 21, 4]));
    close(observed, toWorld([30, 21, 4]));
    report.camera_normals = cameras.map(camera => camera.viewPlaneNormal);
    await expect(page.getByText('Y reconstruction', { exact: true })).toBeVisible();
    await expect(page.getByText('X reconstruction', { exact: true })).toBeVisible();
  });
  await page.getByLabel('Locate (3)', { exact: true }).click();
  await check('Reconstruction clicks retain the same X/Y plane while snapping only acquisition Z to a real layer', async () => {
    for (const [name, target] of [['coronal', [26, 20.5, 3.25]], ['sagittal', [26, 23, 6.25]]]) {
      const before = await state(), camera = before.sliceCameras.find(view => view.name === name).camera;
      const wanted = [...target]; if (name === 'coronal') wanted[1] = before.position[1]; else wanted[0] = before.position[0];
      await clickIndex(name, wanted);
      await expect.poll(async () => (await state()).position[2]).toBe(Math.round(wanted[2]));
      const after = await state(), moved = toWorld(after.position), original = toWorld(wanted);
      assert(Math.abs(dot(camera.viewPlaneNormal, moved.map((value, axis) => value - original[axis]))) < 1e-3);
      wanted[2] = Math.round(wanted[2]); close(after.position, wanted, .01);
    }
  });
  await check('Reconstruction wheel moves exactly one native X or Y voxel and preserves the other coordinates', async () => {
    for (const [name, axis] of [['coronal', 1], ['sagittal', 0]]) {
      const before = await state(), normal = before.sliceCameras.find(view => view.name === name).camera.viewPlaneNormal;
      const basis = before.geometry.direction.slice(axis * 3, axis * 3 + 3), sign = Math.sign(dot(normal, basis));
      const box = await page.getByTestId('viewport-' + name).boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.wheel(0, 120);
      await expect.poll(async () => (await state()).position[axis]).toBeCloseTo(before.position[axis] + sign, 3);
      const expected = [...before.position]; expected[axis] += sign; close((await state()).position, expected, .001);
    }
  });
  await check('Native one-pixel MYO brush / erase and undo operate on the clicked original voxel without leaking to another frame', async () => {
    await page.getByLabel('Expand Acquisition plane', { exact: true }).click(); await page.waitForTimeout(150);
    await page.getByLabel('Select MYO', { exact: true }).click(); await page.getByLabel('Brush (4)', { exact: true }).click();
    const before = await masks(0), revision = (await metadata()).revision;
    const position = (await state()).position, target = [32, 24, position[2]];
    await clickIndex('native', target); await expect.poll(async () => (await metadata()).revision).toBe(revision + 1); await idle();
    const after = await masks(0), offset = volume + (target[2] * height + target[1]) * width + target[0];
    const expected = new Uint8Array(before); expected[offset] = 1; assert.deepEqual(after, expected);
    assert.equal((await masks(1)).reduce((sum, n) => sum + n, 0), 0);
    report.brush_voxel_tzyx = [0, target[2], target[1], target[0]];
    await page.getByLabel('Erase (5)', { exact: true }).click(); await clickIndex('native', target);
    await expect.poll(async () => (await metadata()).revision).toBe(revision + 2); await idle(); assert.deepEqual(await masks(0), before);
    await page.getByLabel('Undo edit', { exact: true }).click(); await idle(); assert.deepEqual(await masks(0), after);
    await page.getByLabel('Expand Acquisition plane', { exact: true }).click();
  });
  await check('Exported 4D labels keep source spacing / origin / direction and the exact original brushed voxel', async () => {
    const outputPath = path.join(privateDir, 'orientation-export.nii.gz');
    fs.writeFileSync(outputPath, await (await get('/export?kind=labelmap')).body());
    const expectedPath = path.join(privateDir, 'orientation-export-expected.json');
    fs.writeFileSync(expectedPath, JSON.stringify({ geometry: study.source_geometry, voxel: report.brush_voxel_tzyx }));
    const result = execFileSync(process.env.CARDIAC_PRISM_TEST_PYTHON || 'python3', ['-c',
      "import json,SimpleITK as s,numpy as n,sys; e=json.load(open(sys.argv[2]));i=s.ReadImage(sys.argv[1]);a=s.GetArrayFromImage(i);g=e['geometry'];assert list(i.GetSize())==g['size'];assert n.allclose(i.GetSpacing(),g['spacing'],atol=1e-5,rtol=0);assert n.allclose(i.GetOrigin(),g['origin'],atol=1e-5,rtol=0);assert n.allclose(i.GetDirection(),g['direction'],atol=1e-5,rtol=0);assert int(a[tuple(e['voxel'])])==2;assert int(n.count_nonzero(a))==1;print(json.dumps({'size':i.GetSize(),'positive_voxels':int(n.count_nonzero(a)),'native_voxel_exact':True}))",
      outputPath, expectedPath], { encoding: 'utf8' });
    report.export = JSON.parse(result);
    assert.deepEqual((await metadata()).source_geometry, study.source_geometry);
  });
  await saveDiagnosticScreenshot(page, { path: path.join(output, 'geometry-candidate.png') }, report);
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) { report.failure = error.stack; process.exitCode = 1; await page.screenshot({ path: path.join(output, 'geometry-failure.png') }).catch(() => {}); }
finally { fs.writeFileSync(path.join(output, 'geometry-browser.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report)); await browser.close(); }
