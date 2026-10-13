import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { base, reportDirectory, browserOptions } from './browser_helpers.mjs';
const baseline = false;
const out = reportDirectory('surface-colors');
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch(browserOptions);
const page = await browser.newPage({ viewport: { width: 360, height: 360 } });
const report = { passed: false, baseline, comparisons: [], identicalGeometry: [], errors: [] };
page.on('pageerror', e => report.errors.push(e.stack || e.message));
const hash = pixels => crypto.createHash('sha256').update(Buffer.from(pixels)).digest('hex');
const previous = !baseline && fs.existsSync(new URL('surface-colors-before.json', out))
  ? JSON.parse(fs.readFileSync(new URL('surface-colors-before.json', out))) : null;
try {
  await page.goto(base + '/tests/surface-colors.html');
  await page.waitForFunction(() => window.surfaceColorsReady, { timeout: 60000 });
  for (const tilted of [false, true]) for (const smooth of [false, true]) for (const color of [0, 1, 2]) for (const side of [1, -1]) {
    const options = { tilted, smooth, color, side };
    const reference = await page.evaluate(options => window.renderSurface({ ...options, reflected: false }), options);
    const reflected = await page.evaluate(options => window.renderSurface({ ...options, reflected: true }), options);
    let delta = 0, brightnessA = 0, brightnessB = 0, foreground = 0;
    for (let p = 0; p < reference.pixels.length; p += 4) {
      if (reference.pixels[p] + reference.pixels[p+1] + reference.pixels[p+2] <= 40) continue;
      foreground++;
      for (let c = 0; c < 3; c++) {
        delta += Math.abs(reference.pixels[p+c] - reflected.pixels[p+c]);
        brightnessA += reference.pixels[p+c]; brightnessB += reflected.pixels[p+c];
      }
    }
    const entry = { ...options, foreground, meanColorDifference: delta / (3 * foreground),
      meanBrightness: brightnessA / (3 * foreground), reflectedBrightness: brightnessB / (3 * foreground),
      boundsDifference: Math.max(...reference.bounds.map((x, i) => Math.abs(x - reflected.bounds[i]))),
      referenceSHA: hash(reference.pixels), reflectedSHA: hash(reflected.pixels) };
    report.comparisons.push(entry);
    assert(foreground > 3000, 'No visible geometry');
    assert(reference.inputUnchanged && reflected.inputUnchanged, 'Mask was modified');
    assert.deepEqual(reference.color, reflected.color);
    assert.deepEqual(reference.material, [.22, .7, .25, 24]);
    assert.deepEqual(reference.material, reflected.material);
    assert.equal(reference.smoothingReady, smooth); assert.equal(reflected.smoothingReady, smooth);
    assert(entry.boundsDifference < (smooth ? .01 : .0001), 'World bounds changed: ' + entry.boundsDifference);
    if (previous) {
      const old = previous.comparisons.find(e => ['tilted', 'smooth', 'color', 'side'].every(k => e[k] === options[k]));
      assert.equal(entry.referenceSHA, old.referenceSHA, 'Original NIfTI rendering changed');
    }
    // Mirroring the voxel lattice can choose a different diagonal on ambiguous
    // marching-cubes cells. Compare lighting without requiring those triangles
    // to shade every pixel identically; unchanged NIfTI pixels are exact above.
    if (!baseline) {
      assert(Math.abs(entry.meanBrightness - entry.reflectedBrightness) / entry.meanBrightness < .03, JSON.stringify(entry));
      assert(entry.meanColorDifference < 20, JSON.stringify(entry));
    }
    if (color === 1 && side === 1) await page.screenshot({ path: new URL(`${baseline ? 'before' : 'after'}-${tilted ? 'oblique' : 'aligned'}-${smooth ? 'smooth' : 'exact'}.png`, out).pathname });
  }
  if (!baseline) for (const tilted of [false, true]) for (const smooth of [false, true]) for (const color of [0, 1, 2]) {
    const options = { tilted, smooth, color, sameLocalGeometry: true };
    const reference = await page.evaluate(options => window.renderSurface({ ...options, reflected: false }), options);
    const reflected = await page.evaluate(options => window.renderSurface({ ...options, reflected: true }), options);
    let delta = 0;
    for (let p = 0; p < reference.pixels.length; p += 4) for (let c = 0; c < 3; c++) delta += Math.abs(reference.pixels[p+c] - reflected.pixels[p+c]);
    const entry = { ...options, meanColorDifference: delta / (reference.pixels.length / 4 * 3) };
    report.identicalGeometry.push(entry);
    // This pair has exactly the same triangle topology and transformed camera;
    // compare actual pixels after compensating the reflected screen handedness.
    assert(entry.meanColorDifference < .1, JSON.stringify(entry));
  }
  assert.deepEqual(report.errors, []);
  if (baseline) assert(report.comparisons.every(e => e.meanColorDifference > 10), 'Baseline did not reproduce dark reflected surfaces');
  report.passed = true;
} catch (e) { report.failure = e.stack; process.exitCode = 1; }
finally {
  fs.writeFileSync(new URL(`surface-colors-${baseline ? 'before' : 'after'}.json`, out), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, comparisons: report.comparisons.length, identicalGeometry: report.identicalGeometry.length,
    worstMeanColorDifference: Math.max(...report.comparisons.map(e => e.meanColorDifference)), failure: report.failure, errors: report.errors }));
  await browser.close();
}
