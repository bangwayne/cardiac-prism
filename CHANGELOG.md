# Changelog

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
