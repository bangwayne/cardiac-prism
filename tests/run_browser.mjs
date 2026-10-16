#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const frontend = path.join(root, 'frontend');
const resultsRoot = path.join(root, 'test-results');
fs.mkdirSync(resultsRoot, { recursive: true });
const output = fs.mkdtempSync(path.join(resultsRoot, 'browser-'));
const workspace = path.join(output, 'workspace');
const python = process.env.CARDIAC_PRISM_TEST_PYTHON ||
  (fs.existsSync(path.join(root, '.venv/bin/python')) ? path.join(root, '.venv/bin/python') : 'python3');
const processes = new Set();
const logs = [];
const report = { passed: false, tests: [] };

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

function start(command, args, environment, name, service = false, cwd = root) {
  const log = fs.openSync(path.join(output, name + '.log'), 'w');
  logs.push(log);
  const child = spawn(command, args, { cwd, env: environment, stdio: ['ignore', 'pipe', 'pipe'] });
  processes.add(child);
  for (const [stream, target] of [[child.stdout, process.stdout], [child.stderr, process.stderr]]) {
    stream.on('data', bytes => { fs.writeSync(log, bytes); if (!service) target.write(bytes); });
  }
  child.finished = new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', (code, signal) => {
      processes.delete(child);
      if (code === 0) resolve();
      else reject(new Error(`${name} exited with ${signal || code}; see ${path.join(output, name + '.log')}`));
    });
  });
  child.finished.catch(() => {});
  return child;
}

async function waitReady(url, child) {
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) await child.finished;
    try { if ((await fetch(url, { signal: AbortSignal.timeout(2000) })).ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error(`Service did not become ready: ${url}`);
}

async function stop(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    try { await child.finished; } catch {}
    return;
  }
  child.kill('SIGTERM');
  const timeout = setTimeout(() => child.kill('SIGKILL'), 5000);
  try { await child.finished; } catch {} finally { clearTimeout(timeout); }
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {
  process.exitCode = 130;
  for (const child of processes) child.kill('SIGTERM');
});

try {
  const apiPort = await freePort(), uiPort = await freePort();
  const api = `http://127.0.0.1:${apiPort}`, ui = `http://127.0.0.1:${uiPort}`;
  const environment = { ...process.env, CARDIAC_PRISM_TEST_OUTPUT: output, CARDIAC_PRISM_TEST_WORKSPACE: workspace,
    CARDIAC_PRISM_TEST_PYTHON: python, CARDIAC_PRISM_WORKSPACE: workspace, CARDIAC_PRISM_PRELOAD: '0', CARDIAC_PRISM_DEVICE: 'cpu',
    CUDA_VISIBLE_DEVICES: '', CARDIAC_PRISM_API_URL: api, CARDIAC_PRISM_URL: ui, OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2' };
  for (const script of ['create_cine_fixture.py', 'make_orientation_fixtures.py']) {
    await start(python, [path.join(root, 'tests', script)], environment, script).finished;
  }
  const backend = start(python, ['-m', 'uvicorn', 'server.app:app', '--host', '127.0.0.1', '--port', String(apiPort)], environment, 'backend', true);
  await waitReady(api + '/api/health', backend);
  const health = await (await fetch(api + '/api/health')).json();
  assert.equal(path.resolve(health.workspace), workspace);
  const vite = start(process.execPath, [path.join(frontend, 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1',
    '--port', String(uiPort), '--strictPort'], environment, 'vite', true, frontend);
  await waitReady(ui, vite);
  for (const script of ['test_browser_artifacts.mjs', 'test_frame_archive.mjs', 'browser_refinement_controls.mjs', 'browser_orientation_pixels.mjs',
    'browser_orientation_geometry.mjs', 'browser_view_orientation_controls.mjs', 'browser_surface_colors.mjs']) {
    console.log(`Running ${script}`);
    const started = Date.now();
    const result = { script, passed: false, seconds: 0 };
    report.tests.push(result);
    try {
      await start(process.execPath, [path.join(root, 'tests', script)], environment, script).finished;
      result.passed = true;
    } catch (error) {
      result.failure = error.stack || String(error);
      throw error;
    } finally {
      result.seconds = (Date.now() - started) / 1000;
    }
  }
  report.passed = true;
} catch (error) {
  report.failure = error.stack;
  process.exitCode = 1;
  console.error(error.message);
} finally {
  await Promise.all([...processes].map(stop));
  for (const descriptor of logs) fs.closeSync(descriptor);
  fs.rmSync(workspace, { recursive: true, force: true });
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`Browser test results: ${output}`);
}
