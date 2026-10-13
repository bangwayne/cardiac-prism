import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';
import { saveDiagnosticScreenshot } from './browser_artifacts.mjs';

function fixture(capture = async () => {}) {
  const page = new EventEmitter();
  page.closed = false;
  page.connected = true;
  page.screenshot = capture;
  page.isClosed = () => page.closed;
  page.context = () => ({ browser: () => ({ isConnected: () => page.connected }) });
  return page;
}

function timeout() {
  return Object.assign(new Error('page.screenshot: Timeout exceeded'), { name: 'TimeoutError' });
}

test('diagnostic screenshot uses a bounded capture and preserves screenshot options', async () => {
  let captured;
  const page = fixture(async options => { captured = options; });
  const report = {};
  assert.equal(await saveDiagnosticScreenshot(page, { path: 'example.png', fullPage: true }, report), true);
  assert.deepEqual(captured, { path: 'example.png', fullPage: true, timeout: 15000 });
  assert.equal(report.artifact_warnings, undefined);
  assert.equal(page.listenerCount('crash'), 0);
});

test('diagnostic screenshot timeout on a healthy page is recorded as a warning', async () => {
  const page = fixture(async () => { throw timeout(); });
  const report = {};
  assert.equal(await saveDiagnosticScreenshot(page, { path: 'example.png' }, report), false);
  assert.deepEqual(report.artifact_warnings, ['Diagnostic screenshot timed out: example.png']);
  assert.equal(page.listenerCount('crash'), 0);
});

test('non-timeout screenshot errors still fail the test', async () => {
  const error = new Error('Screenshot file is not writable');
  const page = fixture(async () => { throw error; });
  const report = {};
  await assert.rejects(saveDiagnosticScreenshot(page, { path: 'example.png' }, report), failure => failure === error);
  assert.equal(report.artifact_warnings, undefined);
  assert.equal(page.listenerCount('crash'), 0);
});

for (const state of ['closed', 'disconnected', 'crashed']) {
  test(`screenshot timeout on a ${state} page still fails the test`, async () => {
    const error = timeout();
    const page = fixture(async () => {
      if (state === 'closed') page.closed = true;
      if (state === 'disconnected') page.connected = false;
      if (state === 'crashed') page.emit('crash');
      throw error;
    });
    const report = {};
    await assert.rejects(saveDiagnosticScreenshot(page, { path: 'example.png' }, report), failure => failure === error);
    assert.equal(report.artifact_warnings, undefined);
    assert.equal(page.listenerCount('crash'), 0);
  });
}
