# Development

## Architecture

Cardiac Prism serves a static React/TypeScript frontend and a FastAPI API from one origin. Cornerstone3D renders linked image and label views; vtk.js builds on-demand mask surfaces. The server stores study files in an explicitly configured workspace and runs the K-Prism model locally.

| Path | Responsibility |
| --- | --- |
| `frontend/src/` | Viewer, cine cache, editing controls, Library, activity and lock UI |
| `server/app.py` | HTTP endpoints, request validation, same-origin frontend and browser headers |
| `server/domain.py`, `server/state.py` | Image geometry, transactions, study state and persistence |
| `server/inference.py`, `server/gpu_pool.py` | Interactive inference, queued batch work and GPU workers |
| `server/cardiac_model.py` | Versioned cardiac preprocessing and checkpoint integration |
| `server/propagation.py` | Same-layer temporal propagation |
| `server/dicom_import.py` | Single-series DICOM validation and conversion |
| `Verse_2_MOE_ui/` | Vendored model architecture and original upstream components |
| `scripts/` | Setup, asset acquisition, startup and verification tools |
| `tests/` | Backend and browser regression tests |
| `assets/` | Versioned asset metadata; large files are downloaded separately |

Source-image and export geometry remain in the original grid. Model tensors are prepared separately. Browser display changes must not become model preprocessing operations.

## Development environment

Install the native environment with development dependencies:

```bash
./scripts/setup.sh --device cpu --dev
```

Use `--device cuda` when testing real GPU inference. Avoid running development against an existing annotation workspace. From the repository root, start a separate instance:

```bash
KPRISM_WORKSPACE="$PWD/workspace-dev" KPRISM_PORT=8507 ./start.sh
```

Build the frontend using the project-local Node runtime or a compatible installed version:

```bash
export PATH="$PWD/runtime/node/bin:$PATH"
cd frontend
npm ci
npm run build
```

The generated frontend is served by the backend on the configured port. A normal production build is the simplest way to exercise the same-origin headers and worker behavior used by installed servers. Do not treat a development server as a deployment artifact.

## Tests

Run the backend suite from the repository root:

```bash
./scripts/test.sh
```

The test helper uses an isolated workspace. Tests generate synthetic fixtures and must not write to a user's study directory. GPU integration checks require the exact released checkpoint and an available GPU; a mocked predictor verifies API behavior but is not evidence of numerical equivalence to the real model.

Run the browser suite with Playwright from the repository root:

```bash
export PATH="$PWD/runtime/node/bin:$PATH"
cd frontend
npx playwright install --with-deps chromium
npm test
```

The runner generates synthetic cine and orientation fixtures, starts temporary backend and frontend servers on unused loopback ports, and checks their workspace before creating studies. It covers editing, 4D playback, orientation pixels, source-space exports, surface colors, and lossless frame caching. It stops the temporary services and removes their study workspace when finished. Reports and screenshots remain under the ignored `test-results/` directory.

The runner uses `.venv/bin/python` by default. Set `KPRISM_TEST_PYTHON` for a different installed environment, or `KPRISM_CHROMIUM_EXECUTABLE` to use an existing compatible Chrome/Chromium executable instead of Playwright's download. These are testing options, not workstation startup settings. Screenshots used in documentation must contain only redistributable or synthetic images.

For frontend changes, run TypeScript compilation and the relevant browser checks. For persistence changes, test both save/restart and full-backup export/import. For changes to inference, compare actual predictions and saved refinement context against the released model under the same preprocessing.

### Real-model HTTP validation

`tests/check_workstation.py` exercises a running test service through its HTTP API. It creates studies and changes the inference device, so use an empty, disposable workspace. With the service above running and the public sample downloaded:

```bash
.venv/bin/python tests/check_workstation.py \
  --url http://127.0.0.1:8507 \
  --output ../cardiac-prism-validation
```

For a GPU installation with two available GPUs, add `--gpu-ids 0,1 --cpu-check --full-cine`. Use the actual GPU IDs returned by `/api/gpus`. The validator maps the first selected GPU to the main inference device using the inventory and `/api/health`; with an unusual visibility mapping, supply the corresponding logical device explicitly, for example `--device cuda:0`. The full check compares every mask and saved probability prior between single-GPU and multiple-GPU inference on the example cine, in addition to editing, locks, propagation, and backup restoration. It can take several minutes and requires space for temporary full-study backups.

After a successful check, restart that test server with the same disposable workspace, then verify persistence:

```bash
.venv/bin/python tests/check_workstation.py \
  --url http://127.0.0.1:8507 \
  --output ../cardiac-prism-validation \
  --resume-report ../cardiac-prism-validation/workstation-validation.json
```

Results and temporary study backups are written to the specified output directory. They are not release source files. A passed API check does not replace browser rendering tests.

## Invariants to preserve

- An interactive click belongs to one time frame, acquisition layer, and structure.
- A slice lock protects masks, prompts, and priors, including against batch work, imports, clearing, and undo/redo.
- Batch jobs check revisions and locks before committing; failure or cancellation leaves the pre-job annotation intact.
- Model input normalization follows the checkpoint's declared preprocessing profile and rejects non-finite values.
- Export preserves source geometry and time dimensions. Overlapping independent masks cannot silently become an ambiguous combined labelmap.
- Surface smoothing affects rendering only. Cine playback hides generated 3D surfaces rather than displaying stale geometry.
- Local edits update the affected state without loading every saved probability prior or invalidating every cached image frame.
- Notes, completion status, and persisted activity are distinct from the in-memory undo/redo stack.

## Licensing changes

Project-authored workstation code uses PolyForm Noncommercial 1.0.0. Vendored upstream components retain their original terms; [NOTICE](../NOTICE) maps them to license texts. Preserve original copyright notices. Do not replace an upstream license header with the project's noncommercial license.

Before adding a dependency or copying code, record its source and license. Regenerate or update third-party notices when shipped frontend dependencies change. Public data and model assets need their own documented provenance and redistribution terms.

## Release checks

1. Install from a clean checkout using only documented dependencies and downloaded assets.
2. Build the frontend and run backend and relevant browser tests.
3. Exercise real CPU and GPU inference with the release checkpoint; verify selected-GPU jobs on supported hardware.
4. Verify the public sample, NIfTI and supported DICOM imports, and 3D/4D navigation.
5. Verify click/brush/erase, lock protection, propagation, save/restart, and full-backup restoration in a temporary workspace.
6. Inspect the release archive for private images, weights outside the asset channel, credentials, local absolute paths, runtime state, and generated test output.
7. Record source version, asset checksums, dependency versions, and the installation paths actually tested.
8. Update the changelog and release documentation. Publish only with the intended repository and asset visibility.

Keep the existing server workspace separate when preparing a release. A new code distribution should not require shipping an internal workstation's cases, notes, or debugging output.
