import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const root = fileURLToPath(new URL('../', import.meta.url));
export const base = process.env.KPRISM_URL;
if (!base || !process.env.KPRISM_TEST_OUTPUT || !process.env.KPRISM_TEST_WORKSPACE) {
  throw new Error('Use npm test from frontend/ to start the isolated browser test services.');
}
export const output = path.resolve(process.env.KPRISM_TEST_OUTPUT);
export const fixturePath = (...parts) => path.join(output, 'fixtures', ...parts);
export function reportDirectory(name) {
  const directory = path.join(output, name);
  fs.mkdirSync(directory, { recursive: true });
  return pathToFileURL(directory + path.sep);
}
export const browserOptions = {
  ...(process.env.KPRISM_CHROMIUM_EXECUTABLE ? { executablePath: process.env.KPRISM_CHROMIUM_EXECUTABLE } : {}),
  headless: true,
  args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
};
export async function assertIsolatedService() {
  const response = await fetch(base + '/api/health');
  assert(response.ok, 'The isolated backend is unavailable.');
  const health = await response.json();
  assert.equal(path.resolve(health.workspace), path.resolve(process.env.KPRISM_TEST_WORKSPACE),
    'Refusing to create test studies outside the isolated test workspace.');
  return health;
}
