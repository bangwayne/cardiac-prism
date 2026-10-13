# Release validation

This record describes installation and behavior checks for version 0.1.0. It verifies software compatibility and preservation of model behavior; it is not an estimate of clinical segmentation accuracy.

## Model equivalence

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

## Application checks

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
