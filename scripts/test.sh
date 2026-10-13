#!/usr/bin/env bash
set -euo pipefail
KPRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KPRISM_PYTHON="${KPRISM_PYTHON:-$KPRISM_ROOT/.venv/bin/python}"
cd "$KPRISM_ROOT"
export PYTHONPATH="$KPRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
KPRISM_TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/cardiac-prism-tests.XXXXXXXX")"
trap 'rm -rf -- "$KPRISM_TEST_TMP"' EXIT
export KPRISM_WORKSPACE="$KPRISM_TEST_TMP/workspace"
export MPLCONFIGDIR="$KPRISM_TEST_TMP/matplotlib"
export KPRISM_PRELOAD=0
export VERSE_DEVICE=cpu
"$KPRISM_PYTHON" -m unittest discover -s tests -p 'test_*.py' -v "$@"

