#!/usr/bin/env bash
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CARDIAC_PRISM_ROOT/scripts/environment.sh"
if [[ -x "$CARDIAC_PRISM_ROOT/runtime/node/bin/node" ]]; then
  export PATH="$CARDIAC_PRISM_ROOT/runtime/node/bin:$PATH"
fi
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo 'Node.js 22 is required. Run ./scripts/setup.sh or install Node.js 22.' >&2
  exit 1
fi
node -e 'if (Number(process.versions.node.split(".")[0]) !== 22) { console.error("Node.js 22 is required."); process.exit(1); }'
cd "$CARDIAC_PRISM_ROOT/frontend"
npm ci --no-audit --no-fund
npm run build

