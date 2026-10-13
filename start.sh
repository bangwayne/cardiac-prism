#!/usr/bin/env bash
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CARDIAC_PRISM_ROOT/scripts/start.sh" "$@"

