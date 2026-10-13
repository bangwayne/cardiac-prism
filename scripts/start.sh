#!/usr/bin/env bash
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CARDIAC_PRISM_ROOT/scripts/environment.sh"
CARDIAC_PRISM_PYTHON="${CARDIAC_PRISM_PYTHON:-$CARDIAC_PRISM_ROOT/.venv/bin/python}"
if [[ ! -x "$CARDIAC_PRISM_PYTHON" ]]; then
  echo 'Python environment missing. Run ./scripts/setup.sh --device cuda (or cpu).' >&2
  exit 1
fi
if [[ ! -f "$CARDIAC_PRISM_ROOT/frontend/dist/index.html" ]]; then
  echo 'Frontend missing. Run ./scripts/build.sh.' >&2
  exit 1
fi
export PYTHONPATH="$CARDIAC_PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CARDIAC_PRISM_CHECKPOINT="${CARDIAC_PRISM_CHECKPOINT:-$CARDIAC_PRISM_ROOT/weight/cardiac-prism-v0.1.pth}"
export CARDIAC_PRISM_WORKSPACE="${CARDIAC_PRISM_WORKSPACE:-$CARDIAC_PRISM_ROOT/workspace}"
export CARDIAC_PRISM_CPU_THREADS="${CARDIAC_PRISM_CPU_THREADS:-2}"
export CARDIAC_PRISM_DEVICE="${CARDIAC_PRISM_DEVICE:-auto}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CARDIAC_PRISM_WORKSPACE/.cache/matplotlib}"
if [[ ! -r "$CARDIAC_PRISM_CHECKPOINT" ]]; then
  echo 'Model weights missing. Run .venv/bin/python scripts/download_assets.py --weights.' >&2
  exit 1
fi
mkdir -p "$CARDIAC_PRISM_WORKSPACE" "$MPLCONFIGDIR"
cd "$CARDIAC_PRISM_ROOT"
"$CARDIAC_PRISM_PYTHON" - <<'PY'
import os
requested = os.environ['CARDIAC_PRISM_DEVICE']
if requested not in ('auto', 'cpu'):
    import torch
    try:
        device = torch.device(requested)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.empty(0, device=device)
    except (ValueError, RuntimeError, AssertionError) as error:
        raise SystemExit(
            f'Cannot start on {requested}: {error}. Check the NVIDIA driver and '
            'Container Toolkit, or select CARDIAC_PRISM_DEVICE=cpu explicitly.'
        ) from error
PY
exec "$CARDIAC_PRISM_PYTHON" -m uvicorn server.app:app \
  --host "${CARDIAC_PRISM_HOST:-127.0.0.1}" --port "${CARDIAC_PRISM_PORT:-8502}" \
  --log-level "${CARDIAC_PRISM_LOG_LEVEL:-info}" "$@"

