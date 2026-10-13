#!/usr/bin/env bash
set -euo pipefail
KPRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$KPRISM_ROOT/runtime/node/bin/node" ]]; then
  export PATH="$KPRISM_ROOT/runtime/node/bin:$PATH"
fi
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo 'Node.js 22 is required. Run ./scripts/setup.sh or install Node.js 22.' >&2
  exit 1
fi
node -e 'if (Number(process.versions.node.split(".")[0]) !== 22) { console.error("Node.js 22 is required."); process.exit(1); }'
cd "$KPRISM_ROOT/frontend"
npm ci --no-audit --no-fund
npm run build

