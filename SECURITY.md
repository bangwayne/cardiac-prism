# Security

## Deployment boundary

Cardiac-Prism is a self-hosted research workstation with no built-in login, per-user authorization, or tenant isolation. Everyone who can reach the API should be treated as a trusted user of the workspace. Editing locks and completion markers are workflow controls, not access controls.

The default listener is `127.0.0.1`. For remote use, connect through an authenticated SSH tunnel. A shared network deployment needs an administrator-managed HTTPS and authentication gateway, appropriate firewall rules, and access restrictions on the study directory. Preserve the application's cross-origin isolation headers and same-origin API routing.

Run one server process per workspace. The service account needs access only to its application files and study storage. Container deployment runs as a non-root user and stores studies in a persistent host directory.

Treat images, notes, exports, backups, and server logs as potentially sensitive. The application is not a DICOM anonymization tool: conversion does not establish that a dataset is safe to share, and identifying information can exist in names or pixel data. Check any example or issue attachment before distribution.

## Model and software integrity

Use the asset downloader's pinned SHA-256 verification. Load only checkpoints from a trusted source; PyTorch checkpoint formats can execute code during deserialization. Do not bypass verification to install an unrelated checkpoint.

Keep the host, browser, container runtime, and dependencies maintained. This release does not claim a clinical certification or a compliance certification. Deployment administrators are responsible for the controls required by their environment.

## Report a vulnerability

Send a private report to the repository maintainer through an established private contact channel. If GitHub private vulnerability reporting is enabled for the repository, use **Security → Report a vulnerability**. If no private channel is available, request one without posting exploit details or sensitive data in an issue.

Include the affected version, deployment method, impact, and a minimal reproduction using synthetic data. Do not include credentials or patient material. The maintainers will coordinate a fix and disclosure; no response-time guarantee is implied.
