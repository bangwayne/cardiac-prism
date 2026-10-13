# Cardiac-Prism Workstation — model assets

For the complete application, screenshots, and installation instructions, open the **[main source branch](https://github.com/bangwayne/cardiac-prism/tree/main)** or the **[v0.1.1 source release](https://github.com/bangwayne/cardiac-prism/tree/v0.1.1)**.

Clone the workstation source explicitly:

```bash
git clone --branch main git@github.com:bangwayne/cardiac-prism.git
cd cardiac-prism
```

Then follow the [installation guide](https://github.com/bangwayne/cardiac-prism/blob/main/docs/install.md) for Docker or native CPU/GPU setup. The installer downloads and verifies the matching weights automatically.

This separate branch stores inference weights with Git LFS. The workstation pins its model download to the immutable `models/v0.1.1` tag. See [model-card.md](model-card.md) for intended use and limitations.

These weights are licensed under PolyForm Noncommercial 1.0.0. Repository access is currently limited to authorized collaborators.
