# Release validation

These checks verify software compatibility and preservation of model behavior; they are not an estimate of clinical segmentation accuracy. Each section records the checks actually performed for that version.

## Version 0.1.2 packaging checks

Version 0.1.2 places the single released checkpoint alongside the source through Git LFS and consolidates license files. The model, preprocessing, and viewer behavior are unchanged from 0.1.1.

- Thirteen asset-download tests, nine release-publication tests, and seven Docker GPU-selection tests pass. They cover downloading from a combined source/model tag and compatibility with older model-only tags.
- Both GitHub workflows pass actionlint validation. CI checks out source without downloading model tensors; release packaging retrieves the checkpoint separately from the same version tag.
- The real checkpoint assembles into seven release attachments with verified checksums. Its Git LFS pointer, size, and SHA-256 match the existing model exactly.
- Third-party license and attribution texts are byte-identical after reorganization. The removed project license copy was identical to the root `LICENSE`.
- A fresh Docker build succeeds with the reorganized files. Model imports, application version, and notice locations pass the container check; all 91 packaged runtime, script, and license file hashes match the source.

## Version 0.1.1 checks

- **280 backend tests** pass in an isolated source-only installation, including legacy configuration compatibility, safe Node.js extraction, resumable release publication, and atomic job completion. The final suite also passes in the constrained Ubuntu container described below.
- Native CPU and NVIDIA Quadro RTX 6000 comparisons each exercise all 18 inference paths and compare 55 arrays with the corresponding same-hardware 0.1.0 baseline. Every array is exactly equal (maximum absolute difference 0). Checkpoint bytes and SHA-256 are unchanged.
- TypeScript compilation and the production frontend build pass with the updated package name and application title.
- All five browser suites pass using the renamed configuration and diagnostic interfaces. They cover refinement controls, shared cine navigation, physical orientation, source-space export, surface colors, and lossless frame caching.
- All five browser suites also pass in an Ubuntu 22.04 container restricted to two CPU cores and 7 GiB RAM, using Chromium software rendering. Readiness waits allow slower rendering without relaxing the exact pixel, geometry, or complete-cine checks.
- Eight deterministic concurrency tests cover completed, failed, and stale outcomes across five segmentation/propagation paths, cancellation during commit, and immediate editing or resubmission. They verify that terminal results cannot be observed before the corresponding study reservation is released. The previous implementation fails the three single-slice outcome regressions.
- A disposable CPU service loads the renamed checkpoint and opens the official ACDC end-diastolic example. Its imported expert labels display in all linked views and the on-demand 3D surface; saving succeeds. The production interface displays **Cardiac-Prism Workstation** and exposes no development diagnostic interface.
- A rebuilt GPU Docker image on a separate NVIDIA Quadro RTX 6000 server also reproduces all 55 baseline arrays exactly. The documented GPU helper verifies the renamed assets, exposes only the selected GPU, opens the 30-frame ACDC cine, runs a real MYO prediction on the requested plane, and saves the study. The service reports version 0.1.1 and the renamed model.
- After the task-completion fix, a rebuilt GPU Docker image passes the real-model HTTP workflow again: positive/negative clicks, one-pixel editing, undo/redo, locked batch segmentation, temporal propagation, and verified study backup/import. The test uses an isolated two-frame, two-layer crop of the public ACDC cine.
- The documentation screenshot was recaptured from the production build using the public ACDC example and reference annotation. GitHub Actions results must be checked separately in the repository; local and remote installation checks do not establish that its hosted workflow succeeded.
- The renamed checkpoint also downloads successfully from the private GitHub model tag using existing Git SSH access and Git LFS, then passes its size and SHA-256 checks.

## Version 0.1.0 installation baseline

### Model equivalence

The release checkpoint contains the same 1,016 model tensors as the workstation checkpoint used to prepare this release. Its file hash differs because private training-path metadata was removed from the checkpoint container. The release checksum and preprocessing profile are recorded in the [model card](model-card.md).

The comparison exercises six inference paths for each of RV, MYO, and LV: organ initialization, clicks only, a negative click, organ plus clicks, one reference, and three references. Each paired run compares the prepared image and all masks, probabilities, and refinement priors: **55 arrays in total**.

| Installation | Compute device | Comparison with the reference implementation |
| --- | --- | --- |
| Fresh native environment | CPU | All 55 arrays exactly equal; maximum absolute difference 0 |
| Fresh native environment | NVIDIA A100 | All 55 arrays exactly equal; maximum absolute difference 0 |
| Fresh native environment | NVIDIA Quadro RTX 6000 | All 55 arrays exactly equal; maximum absolute difference 0 |
| Docker GPU image | NVIDIA A100 | All 55 arrays exactly equal; maximum absolute difference 0 |
| Docker GPU image | NVIDIA Quadro RTX 6000 | All 55 arrays exactly equal; maximum absolute difference 0 |

Each comparison uses the same hardware and input for the reference and release implementation. This does not promise bit-identical floating-point results across different CPU/GPU architectures, library versions, or drivers. The release pins its supported dependency stack to keep those variables explicit.

The A100 Docker comparison also passes when CUDA visibility is specified by a single GPU UUID. Container checks verify that a requested GPU subset is actually exposed, rather than allowing every host GPU.

### Application checks

- **259 backend tests** pass in an isolated release-only source copy without sibling repositories, downloaded weights, or sample data. They cover geometry, inference transactions, click determinism, GPU startup, container GPU selection, local locks, propagation, incremental state, notes, Library operations, activity, and backup restoration.
- **Five browser suites** pass, covering image orientation, 3D surface appearance, editing and navigation controls, and frame-cache behavior.
- The **CPU Docker service** passes a real HTTP workflow using a two-frame, two-layer crop of the public ACDC cine: positive and negative click inference, one-pixel paint priority and erasing, undo/redo, protected batch segmentation, temporal propagation, and full-study backup/import.
- After a **real CPU container restart**, both the working study and its restored backup retain exactly the same masks, 11 saved priors, prompts, notes, slice locks, activity records, and source geometry. Session undo/redo is cleared as documented.
- The **GPU Docker service** passes the same workflow, including switching between CPU and GPU inference. The complete ACDC cine contains **30 time frames × 10 layers = 300 planes**. Single-GPU and two-GPU segmentation produce exactly equal masks and all **900 saved probability priors**, with no overlapping organ labels. Each selected worker processes 150 planes in the two-GPU run.
- After a **real GPU container restart**, the original and restored editing studies retain their masks, 11 priors, prompts, notes, locks, activity, and geometry. The full cine also retains every mask and all 900 priors exactly. Session undo/redo is cleared.

On the tested Quadro RTX 6000 setup with models already loaded, the full-cine segmentation job takes **92.50 seconds with one GPU** and **46.56 seconds with two GPUs**. These are job durations for this input, excluding upload, backup export, and verification; they are not a general performance guarantee.

The full workflow uses actual model predictions. The negative-click result changes pixels only in its intended organ, frame, and acquisition layer. This tests operation and scope; a changed prediction alone is not evidence of improved segmentation accuracy.

## Reproduce the checks

Follow [Development](development.md#tests) for backend and browser checks. The portable [HTTP validation script](../tests/check_workstation.py) exercises the real model and can compare a full cine on one GPU and multiple GPUs. Run it only against a disposable test workspace. Its `--resume-report` option verifies persistence after restarting that test service.

The [ACDC sample](sample-data.md) is a workflow example, not a held-out benchmark claimed for this checkpoint. Performance depends on hardware, image size, browser rendering, and network conditions. No fixed frame rate, runtime, or accuracy is guaranteed for arbitrary inputs.
