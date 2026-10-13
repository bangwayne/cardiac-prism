#!/usr/bin/env bash
set -euo pipefail
KPRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KPRISM_PYTHON="${KPRISM_PYTHON:-$KPRISM_ROOT/.venv/bin/python}"
if [[ ! -x "$KPRISM_PYTHON" ]]; then
  echo 'Python environment missing. Run ./scripts/setup.sh --device cuda (or cpu).' >&2
  exit 1
fi
if [[ ! -f "$KPRISM_ROOT/frontend/dist/index.html" ]]; then
  echo 'Frontend missing. Run ./scripts/build.sh.' >&2
  exit 1
fi
export PYTHONPATH="$KPRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export VERSE_CHECKPOINT="${VERSE_CHECKPOINT:-$KPRISM_ROOT/weight/cardiac_achilleas_epoch10.pth}"
export KPRISM_WORKSPACE="${KPRISM_WORKSPACE:-$KPRISM_ROOT/workspace}"
export KPRISM_CPU_THREADS="${KPRISM_CPU_THREADS:-2}"
export VERSE_DEVICE="${KPRISM_DEVICE:-${VERSE_DEVICE:-auto}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$KPRISM_WORKSPACE/.cache/matplotlib}"
if [[ ! -r "$VERSE_CHECKPOINT" ]]; then
  echo 'Model weights missing. Run .venv/bin/python scripts/download_assets.py --weights.' >&2
  exit 1
fi
mkdir -p "$KPRISM_WORKSPACE" "$MPLCONFIGDIR"
cd "$KPRISM_ROOT"
exec "$KPRISM_PYTHON" -m uvicorn server.app:app \
  --host "${KPRISM_HOST:-127.0.0.1}" --port "${KPRISM_PORT:-8502}" \
  --log-level "${KPRISM_LOG_LEVEL:-info}" "$@"

