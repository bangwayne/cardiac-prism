# Contributing

Contributions that improve annotation reliability, installation, usability, and reproducible research are welcome. Read the [user guide](docs/user-guide.md) and [development guide](docs/development.md) before changing behavior.

## Report an issue

Include the release or commit, installation method, operating system, browser, CPU/GPU choice, steps to reproduce, expected behavior, and observed behavior. If the issue involves image geometry, a small synthetic example is preferable to a clinical image.

Do not attach identifiable patient data, raw clinical DICOM headers, access credentials, or private study backups. Use the process in [SECURITY.md](SECURITY.md) for vulnerabilities.

## Submit a change

1. Create a branch with a focused change.
2. Preserve original spatial geometry, frame/layer scope, and lock protection.
3. Run the checks relevant to the change. Explain test coverage and remaining limitations in the pull request.
4. Update the user or installation guide when the change affects an operation or setup command.
5. Keep generated dependencies, model binaries, cases, runtime state, and test output outside the source commit.

Use clear names and comments that explain a constraint or non-obvious decision. Avoid retaining abandoned code, local deployment notes, or temporary diagnostic prints. A user-facing error should explain what can be corrected without exposing private paths or credentials.

By submitting a contribution, you confirm that you have the right to contribute it under the license applicable to the affected component. Original workstation contributions use PolyForm Noncommercial 1.0.0. Changes to separately licensed upstream components must preserve their license and copyright requirements. Include attribution for any newly incorporated third-party material.
