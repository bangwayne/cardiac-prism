#!/usr/bin/env bash
set -euo pipefail
KPRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$KPRISM_ROOT/scripts/start.sh" "$@"

