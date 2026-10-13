#!/usr/bin/env bash
set -euo pipefail
KPRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-cpu}"
if [[ "$mode" == --help || "$mode" == -h ]]; then
  echo 'Usage: ./scripts/docker.sh cpu|gpu [Compose arguments]'
  echo 'Examples: ./scripts/docker.sh gpu up --build -d; ./scripts/docker.sh gpu logs -f'
  echo 'Set KPRISM_DOCKER_DEVICE_BINDINGS=1 for explicit selected GPU device bindings.'
  exit 0
fi
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo 'First argument must be cpu or gpu.' >&2
  exit 2
fi
if [[ "$#" -gt 0 ]]; then shift; fi
cd "$KPRISM_ROOT"
if ! docker compose version >/dev/null 2>&1; then
  echo 'Docker Engine and the Docker Compose plugin are required.' >&2
  exit 1
fi
export KPRISM_UID="${KPRISM_UID:-$(id -u)}"
export KPRISM_GID="${KPRISM_GID:-$(id -g)}"
compose=(docker compose -f compose.yml)
if [[ "$mode" == gpu ]]; then
  "${KPRISM_DOWNLOAD_PYTHON:-python3}" scripts/docker_devices.py
  compose+=(-f "$KPRISM_ROOT/runtime/compose.gpu-devices.json")
fi
if [[ "$#" == 0 ]]; then set -- up --build -d; fi
if [[ "$1" == up || "$1" == run ]]; then
  mkdir -p workspace weight demo-data/images demo-data/references
  "${KPRISM_DOWNLOAD_PYTHON:-python3}" scripts/download_assets.py --all
fi
exec "${compose[@]}" "$@"

