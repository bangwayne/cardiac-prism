// Rendered-pixel regression, rather than a coordinate round trip alone.
import { chromium, expect } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { base, fixturePath, reportDirectory, browserOptions, assertIsolatedService } from './browser_helpers.mjs';
const output = reportDirectory('orientation-pixels');
const fixtures = JSON.parse(fs.readFileSync(fixturePath('orientation', 'manifest.json'), 'utf8'));
const browser = await chromium.launch(browserOptions);
const page = await browser.newPage({ viewport: { width: 1680, height: 1060 } });
const report = { passed: false, base, checks: [], cases: [], errors: [] };
page.on('pageerror', error => report.errors.push(error.stack || error.message));
const state = () => page.evaluate(() => window.__CARDIAC_PRISM_DIAGNOSTICS__.state());
const intensity = ([x, y, z], t = 0) => 15 + Math.floor(x / 8) * 16 + Math.floor(y / 8) * 35 + Math.floor(z / 3) * 65 + t * 20;
const position = [30, 21, 5];
const viewAxes = { native: 2, coronal: 1, sagittal: 0 };

async function renderedSamples(view, indices, path) {
  const coordinates = await page.evaluate(({ view, indices }) => indices.map(index =>
    window.__CARDIAC_PRISM_DIAGNOSTICS__.indexToCanvas(view, index)), { view, indices });
  const viewport = page.getByTestId('viewport-' + view);
  const image = await viewport.screenshot({ path, style: '.viewport-overlay { visibility: hidden !important; }' });
  return page.evaluate(async ({ encoded, coordinates }) => {
    const picture = new Image(); picture.src = 'data:image/png;base64,' + encoded; await picture.decode();
    const canvas = document.createElement('canvas'); canvas.width = picture.width; canvas.height = picture.height;
    const context = canvas.getContext('2d', { willReadFrequently: true }); context.drawImage(picture, 0, 0);
    return coordinates.map(point => {
      const x = Math.round(point[0]), y = Math.round(point[1]);
      if (x < 1 || x >= canvas.width - 1 || y < 1 || y >= canvas.height - 1) return { point, outside: true };
      const bytes = context.getImageData(x - 1, y - 1, 3, 3).data;
      return { point, rgb: [0, 1, 2].map(channel => Array.from({ length: 9 }, (_, n) => bytes[n * 4 + channel])
        .sort((a, b) => a - b)[4]) };
    });
  }, { encoded: image.toString('base64'), coordinates });
}

async function assertPixels(record, time) {
  for (const [view, axis] of Object.entries(viewAxes)) {
    const title = view === 'native' ? 'Acquisition plane' : record.aligned ?
      (view === 'coronal' ? 'Coronal' : 'Sagittal') : (view === 'coronal' ? 'Y reconstruction' : 'X reconstruction');
    await page.getByLabel('Expand ' + title, { exact: true }).click();
    await page.getByLabel('Fit ' + title, { exact: true }).click();
    await page.waitForTimeout(180);
    const current = await state();
    const fixed = current.position;
    const indices = [[5, 5, 1], [13, 13, 4], [21, 29, 7], [45, 37, 10]].map(index => {
      index[axis] = fixed[axis]; return index;
    });
    const samples = await renderedSamples(view, indices, new URL(`${record.name}-${view}-t${time}.png`, output).pathname);
    const low = current.window.lower, high = current.window.upper;
    samples.forEach((sample, index) => {
      assert(!sample.outside, `${record.name}/${view}: source voxel is outside the fitted viewport`);
      const value = intensity(indices[index], time), expected = Math.max(0, Math.min(255, (value - low) / (high - low) * 255));
      const error = Math.max(...sample.rgb.map(component => Math.abs(component - expected)));
      assert(error < 9, `${record.name}/${view}/t${time} voxel ${indices[index]} expected ${expected}, saw ${sample.rgb}`);
      record.pixels.push({ view, time, index: indices[index], value, expected_gray: +expected.toFixed(3), rgb: sample.rgb });
    });
    // A view's normal is exactly one acquisition axis, even for det=-1 scans.
    const normal = current.sliceCameras.find(camera => camera.name === view).camera.viewPlaneNormal;
    const basis = record.direction.filter((_, offset) => offset % 3 === axis);
    assert(Math.abs(Math.abs(normal.reduce((sum, n, i) => sum + n * basis[i], 0)) - 1) < 1e-5);
    await page.getByLabel('Expand ' + title, { exact: true }).click();
    await page.waitForTimeout(100);
  }
}

try {
  await assertIsolatedService();
  for (const fixture of fixtures) {
    const response = await page.request.post(base + '/api/projects', { multipart: { file: {
      name: fixture.name + '.nii.gz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(fixture.path),
    } } });
    assert(response.ok(), await response.text());
    const study = await response.json();
    const record = { ...fixture, study_id: study.id, aligned: fixture.name.endsWith('-aligned'), pixels: [] };
    report.cases.push(record);
    await page.addInitScript(({ id, position }) => localStorage.setItem('kprism-view-' + id,
      JSON.stringify({ tool: 'crosshair', position })), { id: study.id, position });
    await page.goto(base + `/?study=${study.id}&debug=1`);
    await expect(page.getByText('Viewer ready', { exact: true })).toBeVisible({ timeout: 60000 });
    const initial = await state();
    assert.deepEqual(initial.position, position);
    assert.equal(initial.reconstructionGeometry.convention, 'acquisition');
    assert.equal(initial.reconstructionGeometry.patientAxesAligned, record.aligned);
    const normals = initial.sliceCameras.map(item => item.camera.viewPlaneNormal);
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++)
      assert(Math.abs(normals[i].reduce((sum, n, k) => sum + n * normals[j][k], 0)) < 1e-5);
    await assertPixels(record, 0);
    await page.getByLabel('Time frame', { exact: true }).focus();
    await page.getByLabel('Time frame', { exact: true }).press('End');
    await expect.poll(async () => (await state()).time).toBe(1);
    await assertPixels(record, 1);
    await page.screenshot({ path: new URL(fixture.name + '-grid.png', output).pathname });
    report.checks.push(`${fixture.name}: all three rendered planes show the correct source voxels in both time frames`);
  }
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack; process.exitCode = 1;
  await page.screenshot({ path: new URL('pixels-failure.png', output).pathname }).catch(() => {});
} finally {
  fs.writeFileSync(new URL('rendered-pixels.json', output), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors, failure: report.failure }));
  await browser.close();
}
