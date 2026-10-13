#!/usr/bin/env bash
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CARDIAC_PRISM_ROOT/scripts/environment.sh"
mode="${1:-cpu}"
if [[ "$mode" == --help || "$mode" == -h ]]; then
  echo 'Usage: ./scripts/docker.sh cpu|gpu [Compose arguments]'
  echo 'Examples: ./scripts/docker.sh gpu up --build -d; ./scripts/docker.sh gpu logs -f'
  echo 'Set CARDIAC_PRISM_DOCKER_DEVICE_BINDINGS=1 for explicit selected GPU device bindings.'
  exit 0
fi
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo 'First argument must be cpu or gpu.' >&2
  exit 2
fi
if [[ "$#" -gt 0 ]]; then shift; fi
cd "$CARDIAC_PRISM_ROOT"
if ! docker compose version >/dev/null 2>&1; then
  echo 'Docker Engine and the Docker Compose plugin are required.' >&2
  exit 1
fi
export CARDIAC_PRISM_UID="${CARDIAC_PRISM_UID:-$(id -u)}"
export CARDIAC_PRISM_GID="${CARDIAC_PRISM_GID:-$(id -g)}"
compose=(docker compose -f compose.yml)
if [[ "$mode" == gpu ]]; then
  "${CARDIAC_PRISM_DOWNLOAD_PYTHON:-python3}" scripts/docker_devices.py
  compose+=(-f "$CARDIAC_PRISM_ROOT/runtime/compose.gpu-devices.json")
fi
if [[ "$#" == 0 ]]; then set -- up --build -d; fi
if [[ "$1" == up || "$1" == run ]]; then
  mkdir -p workspace weight demo-data/images demo-data/references
  "${CARDIAC_PRISM_DOWNLOAD_PYTHON:-python3}" scripts/download_assets.py --all
fi
exec "${compose[@]}" "$@"

