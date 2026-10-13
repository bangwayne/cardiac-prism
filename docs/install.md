# Installation and administration

This guide installs a standalone Cardiac-Prism server with the released cardiac model. All viewing, editing, and inference use your own server. A remote browser connects through an SSH tunnel or a suitably configured HTTPS reverse proxy.

## 1. Choose an installation

| Method | Use it when | Requirements |
| --- | --- | --- |
| Docker, NVIDIA GPU | Docker is available and interactive inference matters | Linux x86-64, Docker Engine with Compose, NVIDIA driver, NVIDIA Container Toolkit |
| Docker, CPU | There is no supported NVIDIA GPU | Linux x86-64, Docker Engine with Compose |
| Native Python | Containers are unavailable or you need a development environment | Linux x86-64, Python 3.10, Git, C/C++ compiler and build tools; NVIDIA driver for GPU use |

The browser can run on Linux, macOS, or Windows. Use a current Chrome or Edge with hardware acceleration and WebGL enabled. The GPU in the **server** performs model inference; the GPU in the **browser computer** draws the viewer. A fast server alone does not fix slow browser rendering.

Plan disk space for dependencies, the checkpoint, source images, editable masks, and backups. Large 4D studies need more RAM and storage than the checkpoint itself. A GPU with at least 8 GB of available memory and a server with at least 16 GB RAM are useful starting points, not guaranteed limits for every study. Resource use depends on image size, number of resident studies, and simultaneous GPU workers.

The release uses a pinned PyTorch/CUDA stack. Installing a newer CUDA toolkit on the host does not change the bundled runtime. Verify that your NVIDIA driver supports the packaged runtime and that your GPU architecture is supported by the selected PyTorch build.

## 2. Download the source and model

The repository is currently private. Ask the maintainer to grant your GitHub account access, configure that account's SSH key on the server, and install Git LFS. For a Debian/Ubuntu system, Git and Git LFS can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install
```

Check GitHub SSH authentication with `ssh -T git@github.com`. GitHub's successful authentication message states that shell access is unavailable; that is expected. Use an SSH clone so the asset downloader can reuse the same authorized access:

```bash
git clone --branch main git@github.com:bangwayne/cardiac-prism.git
cd cardiac-prism
```

For a reproducible installation, select a published release tag instead of a moving development branch:

```bash
git checkout v0.1.2
```

The `main` branch and `v0.1.2` source tag contain both the application and the released model. Git LFS downloads `weight/cardiac-prism-v0.1.pth` during a normal clone; its matching JSON profile is tracked beside it. You do not need a second branch or repository. The public ACDC example is downloaded during setup. All asset sizes and SHA-256 values are pinned in the manifest.

If cloning skipped the model download, run `git lfs pull` or `python3 scripts/download_assets.py --weights`. The downloader can use the GitHub Release or recover the same checkpoint through Git LFS at the immutable `v0.1.2` source tag. It reuses the existing Git remote and SSH authentication; no token needs to be written to a configuration file. Do not substitute a different checkpoint while retaining this release's model description: the architecture, preprocessing, and weights must agree.

## 3. Install with Docker

Install [Docker Engine](https://docs.docker.com/engine/install/) and the [Compose plugin](https://docs.docker.com/compose/install/linux/) using the instructions for your distribution. The `docker` command must work for your account. The helper also needs Python 3.9 or later, Git LFS, and authorized GitHub SSH access on the host for private asset downloads. These credentials do not need to be copied into the container.

Keep a Docker installation and its persistent directories on a filesystem the Docker daemon can access. On shared research servers, home/project directories can be NFS mounts that deny the daemon access even though your account can read them. In that case, clone the project into an administrator-provisioned **local, durable** directory such as `/srv/projects/cardiac-prism`, or use the native installation. Do not keep production studies under `/tmp`, which can be cleaned automatically, and do not change broad home-directory permissions to work around a mount error.

For NVIDIA GPUs, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Confirm `nvidia-smi` works on the host before proceeding.

### GPU server

```bash
./scripts/docker.sh gpu up --build -d
./scripts/docker.sh gpu logs --tail=100 -f
```

Press `Ctrl+C` to stop following the logs; the detached service continues running.

### CPU server

```bash
./scripts/docker.sh cpu up --build -d
./scripts/docker.sh cpu logs --tail=100 -f
```

The helper prepares the writable host directories, downloads and verifies assets, builds the frontend and Python image, and starts the service as a non-root user. Model and example directories are mounted read-only; the workspace is writable and survives container replacement.

### Select GPUs

The GPU container exposes the configured NVIDIA devices. Choose the inference GPU through **Inference device** and the GPUs for a batch job through the multiple-GPU selector. GPU indices inside a container can differ from host indices; use the devices actually offered by the running workstation.

To restrict which host GPUs are exposed, set `NVIDIA_VISIBLE_DEVICES` to physical host indices or complete GPU UUIDs before starting:

```bash
NVIDIA_VISIBLE_DEVICES=0,1 ./scripts/docker.sh gpu up --build -d
```

Use the workstation's displayed IDs after startup; they can be renumbered inside the container.

Run either the CPU or GPU service on a given host port. If switching variants, save your work, stop the existing variant, then start the other:

```bash
./scripts/docker.sh cpu down
./scripts/docker.sh gpu up --build -d
```

Stopping a service discards the session undo/redo stack. Saved studies remain in the workspace.

## 4. Install without Docker

You need Python **3.10**, its virtual-environment and development support, Git, and a working C/C++ toolchain. For example, on a distribution that provides Python 3.10 packages:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3.10 python3.10-venv python3.10-dev
```

Package names vary by distribution. On a server with Conda, an isolated Python bootstrap is an alternative:

```bash
conda create -n cardiac-prism-bootstrap python=3.10 -y
conda activate cardiac-prism-bootstrap
CARDIAC_PRISM_BOOTSTRAP_PYTHON=python ./scripts/setup.sh --device cuda
```

Choose one setup command:

```bash
./scripts/setup.sh --device cuda
```

or:

```bash
./scripts/setup.sh --device cpu
```

Setup creates `.venv`, installs the pinned Python stack and model dependencies, installs a project-local Node runtime if required, runs `npm ci`, builds the frontend, prepares dcm2niix, verifies the checkpoint, and downloads the example and any missing assets. It does not need administrator access after the system prerequisites are available. Native inference does not require a separate host CUDA toolkit when using the packaged PyTorch wheels.

Start the application from the repository root:

```bash
./start.sh
```

Keep the terminal open while using it. `Ctrl+C` stops the service. For longer sessions, use your institution's service manager or a terminal multiplexer such as `tmux`.

### Choose an inference device

Native startup selects a GPU when available, otherwise CPU. To choose explicitly:

```bash
CARDIAC_PRISM_DEVICE=cpu ./start.sh
```

```bash
CARDIAC_PRISM_DEVICE=cuda:0 ./start.sh
```

The UI can switch among available devices while no inference job is running. **Segment all time frames** and temporal propagation can use several selected GPUs. Each selected GPU loads its own model, so the first job includes loading time. Results are applied only after the entire job succeeds; the progress panel shows preparation, prediction, and application phases.

### Native configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `CARDIAC_PRISM_HOST` | Listening address | `127.0.0.1` |
| `CARDIAC_PRISM_PORT` | HTTP port | `8502` |
| `CARDIAC_PRISM_DEVICE` | `auto`, `cpu`, or `cuda:N` | `auto` |
| `CARDIAC_PRISM_CHECKPOINT` | Checkpoint file; keep its matching JSON profile alongside it | `weight/cardiac-prism-v0.1.pth` |
| `CARDIAC_PRISM_CONFIG` | Model architecture and inference configuration | `cardiac_prism/config/eval_config.yaml` |
| `CARDIAC_PRISM_WORKSPACE` | Persistent study and library directory | `workspace/` under the repository |
| `CARDIAC_PRISM_PYTHON` | Interpreter used by `start.sh` | Project `.venv` Python |
| `CARDIAC_PRISM_BOOTSTRAP_PYTHON` | Python used to create the environment | `python3.10` |
| `CARDIAC_PRISM_VENV` | Environment directory created by setup | `.venv/` |
| `CARDIAC_PRISM_DCM2NIIX` | Path to a separately installed DICOM converter | Project-local converter |

If you create a nondefault environment with `CARDIAC_PRISM_VENV`, set `CARDIAC_PRISM_PYTHON` to that environment's Python when starting. Use the `CARDIAC_PRISM_` names for new configurations. Legacy `KPRISM_` settings remain accepted for existing installations, with the new name taking precedence when both are set; legacy `VERSE_DEVICE`, `VERSE_CONFIG`, and `VERSE_CHECKPOINT` map to `CARDIAC_PRISM_DEVICE`, `CARDIAC_PRISM_CONFIG`, and `CARDIAC_PRISM_CHECKPOINT`.

Example with study storage on another disk:

```bash
CARDIAC_PRISM_WORKSPACE=/data/cardiac-prism/studies CARDIAC_PRISM_PORT=8502 ./start.sh
```

The service account must have read/write access to this directory. Run only one application process per workspace; this release is not a distributed database service. Do not add multiple Uvicorn workers against the same workspace.

## 5. Connect from a Mac or Windows computer

Run this **on your own computer**, not inside an existing server SSH session:

```bash
ssh -N -L 8502:127.0.0.1:8502 YOUR_USERNAME@YOUR_SERVER
```

On macOS, use Terminal. On Windows, use PowerShell or Windows Terminal with the OpenSSH client installed. Replace the username and server name with your account details, authenticate as usual, and leave the terminal open.

Open **http://localhost:8502/** in Chrome or Edge. Use **http://localhost:8502/?library=1** to open the Library directly.

If your computer already uses port 8502:

```bash
ssh -N -L 8512:127.0.0.1:8502 YOUR_USERNAME@YOUR_SERVER
```

Then open **http://localhost:8512/**. The first port in `-L` belongs to your computer; the last port belongs to the server. Changing only the browser address does not create a tunnel.

The same approach works with an editor's port-forwarding panel: forward the server's **8502**, then open the local address reported by the editor.

### HTTPS reverse proxies

The viewer requires a secure browser context for shared-memory and rendering features. `localhost` through SSH is supported. For a domain, use HTTPS and retain the application's `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. Proxy the frontend and `/api/` from the same origin. Allow sufficiently large request bodies for uploads and appropriate timeouts for large studies.

There is no built-in authentication, user separation, or access-control system. Place an authentication gateway in front of a network-accessible service. A study's completion marker or slice lock is an editing aid, not a security boundary. See [SECURITY.md](../SECURITY.md).

## 6. Confirm installation

Check the server:

```bash
curl --fail http://127.0.0.1:8502/api/health
python3 scripts/download_assets.py --all --verify
```

The health endpoint can respond before a lazily loaded model has been initialized. A successful health request alone does not prove GPU inference is working.

In the browser:

1. Open the ACDC cine example from **Library**.
2. Move between layers and time frames; play and pause the cine.
3. Select **MYO** and run **Segment this slice**. Confirm a mask appears.
4. Add a positive or negative click, then make and undo a small brush stroke.
5. Use **Render 3D** while paused. Start playback and confirm the surface is hidden.
6. Click **Save study**, export **Full study backup**, and reopen the saved study from the Library.

The detailed [user guide](user-guide.md) explains every operation and its scope.

## 7. Model and example files

```bash
python3 scripts/download_assets.py --weights
python3 scripts/download_assets.py --sample
python3 scripts/download_assets.py --all --verify
```

The downloader validates pinned SHA-256 digests and avoids using an incomplete download as a completed asset. To use a copy of the exact released checkpoint already present on your server:

```bash
python3 scripts/download_assets.py --weights-file /path/to/cardiac-prism-v0.1.pth
```

This option validates the checkpoint against the release manifest; it is not a general checkpoint conversion tool. See the [model card](model-card.md) and [sample-data guide](sample-data.md).

For installations with restricted network access, download the assets on a permitted machine and transfer the verified files to their documented locations. Native `./scripts/setup.sh --device cuda --skip-assets` skips the asset download, but still needs access to Python/Node dependencies unless those packages have also been mirrored or cached. It is not a fully offline installer.

## 8. Save, back up, and upgrade

Use **Save study** after editing. The browser's **Unsaved edits** indicator distinguishes annotation changes from saved state. Notes and lock metadata have separate persistence; saving a note does not save every pending annotation change.

For a portable case, choose **Full study backup** in **Export format**. This includes the source image and working annotation state. Ordinary NIfTI exports carry masks and geometry, but do not preserve all study metadata or future click-refinement context. Neither backup nor restart restores the current session's undo/redo stack.

For a server-wide backup:

1. Ask connected users to save and stop editing; wait for or cancel active jobs.
2. Stop the service gracefully.
3. Copy the entire workspace directory to separate storage. Keep the release version and asset manifest alongside the backup.
4. Restart the service using the same workspace.

Do not copy a live workspace while annotations are being written and assume the copy is a consistent backup.

Before upgrading, back up the workspace and read [CHANGELOG.md](../CHANGELOG.md). Check out the desired release, rerun setup or rebuild the container, verify assets, and start against the preserved workspace. Do not delete the workspace, replace it with a sample workspace, or use a container cleanup command that deletes your study storage.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Address already in use` | A server or SSH tunnel already owns that port. Reuse it or choose a different local port; do not start duplicate services against one workspace. |
| Browser cannot connect | Confirm the service is running, the SSH terminal remains open, and the browser port matches the first port in `-L`. |
| Viewer is blank or reports a shared-memory error | Use localhost through SSH or HTTPS; retain isolation headers, enable browser hardware acceleration, and inspect the browser's WebGL availability. |
| No GPU is offered | Run `nvidia-smi`; check Docker NVIDIA runtime configuration or the CUDA-enabled native environment and visible-device settings. |
| GPU container reports `Failed to initialize NVML: Unknown Error` | See the explicit device-binding workaround below, or use the native installation. |
| First segmentation is slow | The first request loads a model. A multiple-GPU job loads one copy per selected GPU. Watch the job's preparation and per-GPU progress. |
| A large job shows progress but no partial masks | Whole-study jobs apply their masks together when complete. Cancel keeps the current annotation. |
| CUDA out of memory | Reduce selected workers, free memory occupied by other jobs, select another GPU, or use CPU. Keep at least one verified backup before restarting. |
| Cine is slower than the selected FPS | FPS is a target. Browser GPU, transfer speed, first-pass loading, and study size affect actual playback. Keep 3D surfaces hidden while playing. |
| An annotation will not import | Check dimensions, time dimension, origin, spacing, and direction against the source; matching pixel counts alone are insufficient. |
| DICOM folder is rejected | Confirm it contains one uncompressed classic MR series with a consistent and complete frame/layer grid. See supported formats in the user guide. |
| Download verification fails | Remove only the failed asset and retry the downloader. Do not disable digest verification. |
| Private model download fails | Confirm your account has repository access, `ssh -T git@github.com` authenticates, `git lfs version` works, and `git remote -v` shows the authorized SSH remote. |
| Editing is disabled | Pause playback, wait for a job, reopen a completed study if required, and inspect organ and slice locks. |

For a reproducible issue, include the release version, installation method, browser, device type, and a de-identified error description. Avoid posting patient images, study backups, access tokens, or raw DICOM headers in public issues.

### NVIDIA container device access

Some hosts have a [documented NVIDIA Container Toolkit/cgroup device-access issue](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error). The host's `nvidia-smi` may work while a container reports an NVML error or cannot initialize CUDA. Cardiac-Prism's helper offers explicit device bindings for this situation:

```bash
CARDIAC_PRISM_DOCKER_DEVICE_BINDINGS=1 NVIDIA_VISIBLE_DEVICES=0,1 \
  ./scripts/docker.sh gpu up -d
```

Use the physical host GPU indices or complete UUIDs assigned to your workload. The helper binds those GPUs and the required NVIDIA control/UVM devices; it does not enable privileged mode or reconfigure the Docker daemon. The generated override is kept in the ignored `runtime/` directory. For later starts, keep both variables in the shell environment or repeat the command prefix. This helper option is read before Compose loads `.env`, so setting it only in `.env` is insufficient. This addresses device exposure; it does not install a missing driver or make an unsupported GPU architecture compatible.
