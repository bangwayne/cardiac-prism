# Changelog

## Unreleased

- Clarify that the released workstation and checkpoint support LV, RV, and MYO segmentation in cardiac cine MRI; link upstream K-Prism training and welcome community model/data integrations.
- Document the ACDC, M&Ms, and M&Ms-2 fine-tuning cohort, split policy, label mapping, and training recipe.
- Add the VerSe IPMI 2025 reference alongside K-Prism in the research citation instructions and machine-readable citation metadata.
- Use public HTTPS installation instructions and consolidate upstream model component licenses into one file while preserving the original texts.

## 0.1.2

- Keep application code and the single released cardiac checkpoint together on `main`; Git LFS stores the unchanged checkpoint.
- Pin source and model downloads to the same release tag. Retain older model tags for existing installations.
- Keep the project license at the repository root and consolidate upstream notices under `licenses/`, preserving their complete terms and attribution.

## 0.1.1

- Standardize the application name as Cardiac-Prism Workstation across the interface, documentation, exports, and package metadata.
- Package the model architecture under `cardiac_prism/` and rename the unchanged v0.1 checkpoint and profile to `cardiac-prism-v0.1.pth` and `cardiac-prism-v0.1.json`.
- Use `CARDIAC_PRISM_` configuration names, retaining compatibility with existing environment settings and saved browser preferences.
- Document GitHub-hosted model downloads through Release assets and the versioned Git LFS asset tag; no Hugging Face account is needed.
- Resume interrupted GitHub Release drafts, verify uploaded attachments before publishing, and retain immutable source and model tags.
- Fix Node.js installation on Ubuntu Python builds whose archive filter rejects npm executable links; validate and create those links separately.
- Publish inference and propagation results only after releasing the study's editing reservation, so immediate edits or undo no longer race with task completion. Preserve safe cancellation when a new task is queued.
- Make browser checks tolerate slower software rendering while still checking every cine frame and exact image/mask values; bound test-only CPU thread usage on small runners.

## 0.1.0

Initial packaged workstation release.

- Linked acquisition/reconstruction views, shared cine time cursor, and an on-demand 3D mask view.
- Scalar 3D/4D NIfTI and single-series classic MR DICOM folder imports with geometry and timing validation.
- RV, MYO, and LV segmentation with positive/negative clicks, one-pixel brush/erase, and probability-based overlap resolution for joint prediction.
- Whole-frame and whole-cine segmentation with CPU, single-GPU, and selected multiple-GPU execution.
- Same-layer temporal propagation from one to three reference frames.
- Frame/layer slice locks, organ locks, notes, completion states, activity history, and Library management.
- Persistent study storage, NIfTI annotation exchange, and portable full-study backup/restore.
- Native and Docker installation paths, pinned model and example downloads, integrity checks, and English installation and operation guides.
- Noncommercial workstation/model licensing with separate upstream code and sample-data notices.

Current limits: Linux x86-64 server target; no built-in authentication; one process per workspace; session undo/redo is not restored after a restart; no compressed or enhanced multi-frame DICOM; no automatic propagation of an interactive click beyond its current 2D frame/layer.
