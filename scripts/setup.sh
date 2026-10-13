#!/usr/bin/env bash
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CARDIAC_PRISM_ROOT/scripts/environment.sh"
device=cpu
skip_assets=0
with_dev=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --device)
      if [[ "${2:-}" != cpu && "${2:-}" != cuda ]]; then
        echo '--device requires cpu or cuda.' >&2
        exit 2
      fi
      device="$2"; shift 2 ;;
    --skip-assets) skip_assets=1; shift ;;
    --dev) with_dev=1; shift ;;
    --help|-h)
      echo 'Usage: ./scripts/setup.sh [--device cpu|cuda] [--dev] [--skip-assets]'
      echo 'Requires Linux x86_64, Python 3.10 with venv, Git and a C++ compiler.'
      echo 'CARDIAC_PRISM_BOOTSTRAP_PYTHON selects Python; CARDIAC_PRISM_VENV selects the installation directory.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
CARDIAC_PRISM_BOOTSTRAP_PYTHON="${CARDIAC_PRISM_BOOTSTRAP_PYTHON:-python3.10}"
CARDIAC_PRISM_VENV="${CARDIAC_PRISM_VENV:-$CARDIAC_PRISM_ROOT/.venv}"
"$CARDIAC_PRISM_BOOTSTRAP_PYTHON" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Python 3.10 is required to match the released model environment.')
if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
    raise SystemExit('This release supports Linux x86_64 servers. Use a browser on macOS/Windows.')
PY
for executable in git c++; do
  if ! command -v "$executable" >/dev/null; then
    echo "Missing $executable. On Ubuntu/Debian install git build-essential python3.10-venv python3.10-dev." >&2
    exit 1
  fi
done
if [[ ! -x "$CARDIAC_PRISM_VENV/bin/python" ]]; then
  "$CARDIAC_PRISM_BOOTSTRAP_PYTHON" -m venv "$CARDIAC_PRISM_VENV"
fi
CARDIAC_PRISM_PYTHON="$CARDIAC_PRISM_VENV/bin/python"
"$CARDIAC_PRISM_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else "The existing virtual environment must use Python 3.10.")'
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CARDIAC_PRISM_ROOT/runtime/pip-cache}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PYTHONDONTWRITEBYTECODE=1
cd "$CARDIAC_PRISM_ROOT"
"$CARDIAC_PRISM_PYTHON" -m pip install -r requirements-build.txt
variant=cpu
if [[ "$device" == cuda ]]; then variant=cu117; fi
"$CARDIAC_PRISM_PYTHON" -m pip install \
  "torch==1.13.1+$variant" "torchvision==0.14.1+$variant" \
  -c requirements-lock.txt \
  --extra-index-url "https://download.pytorch.org/whl/$variant"
"$CARDIAC_PRISM_PYTHON" -m pip install -r requirements-model.txt
# The workstation uses standard Torch layers, not detectron2's CUDA operators.
# Building the CPU extensions makes installation independent of a CUDA toolkit.
CUDA_VISIBLE_DEVICES='' FORCE_CUDA=0 "$CARDIAC_PRISM_PYTHON" -m pip install \
  --no-build-isolation --no-deps \
  'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@0ae803b1449cd2d3f8fa1b7c0f59356db10b3083'
if [[ "$with_dev" == 1 ]]; then
  "$CARDIAC_PRISM_PYTHON" -m pip install -r requirements-dev.txt
fi
"$CARDIAC_PRISM_PYTHON" -m pip check
"$CARDIAC_PRISM_PYTHON" "$CARDIAC_PRISM_ROOT/scripts/setup_node.py"
CARDIAC_PRISM_PYTHON="$CARDIAC_PRISM_PYTHON" "$CARDIAC_PRISM_ROOT/scripts/setup-dicom.sh"
"$CARDIAC_PRISM_ROOT/scripts/build.sh"
if [[ "$skip_assets" == 0 ]]; then
  "$CARDIAC_PRISM_PYTHON" "$CARDIAC_PRISM_ROOT/scripts/download_assets.py" --all
fi
"$CARDIAC_PRISM_PYTHON" "$CARDIAC_PRISM_ROOT/scripts/check_environment.py" --device "$device"
echo 'Installation complete. Start with ./start.sh; open http://127.0.0.1:8502/.'
if [[ "$CARDIAC_PRISM_VENV" != "$CARDIAC_PRISM_ROOT/.venv" ]]; then
  echo "For this custom environment, set CARDIAC_PRISM_PYTHON=$CARDIAC_PRISM_PYTHON before starting."
fi

