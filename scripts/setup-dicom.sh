#!/usr/bin/env bash
# Install the verified Linux x86-64 converter for DICOM folder import.
set -euo pipefail
CARDIAC_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CARDIAC_PRISM_ROOT/scripts/environment.sh"
CARDIAC_PRISM_PYTHON="${CARDIAC_PRISM_PYTHON:-$CARDIAC_PRISM_ROOT/.venv/bin/python}"
if [[ ! -x "$CARDIAC_PRISM_PYTHON" ]]; then CARDIAC_PRISM_PYTHON=python3; fi
"$CARDIAC_PRISM_PYTHON" - "$CARDIAC_PRISM_ROOT" <<'PY'
import hashlib
import io
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.request
import zipfile

root = Path(sys.argv[1])
version = 'v1.0.20260724'
archive_sha = '2916a7bc64511eb20fb976a994799183cc1bfdd03395bf5d8edef54a58dee06f'
binary_sha = '32c43ae00a46458ebdaea7585f2cf4bc0354734423faba8844a15971295e93e0'
folder = root / 'runtime' / 'tools' / ('dcm2niix-' + version)
target = folder / 'dcm2niix'
if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == binary_sha:
    target.chmod(0o750)
    print('Verified dcm2niix ' + version + ': ' + str(target))
    raise SystemExit(0)
if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
    raise SystemExit('Install dcm2niix for this server and set CARDIAC_PRISM_DCM2NIIX to its executable path.')
url = 'https://github.com/rordenlab/dcm2niix/releases/download/' + version + '/dcm2niix_lnx.zip'
with urllib.request.urlopen(url, timeout=60) as response:
    payload = response.read(16 * 1024 * 1024 + 1)
if len(payload) > 16 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != archive_sha:
    raise SystemExit('Converter download failed its size or SHA-256 check; nothing installed.')
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    candidates = [item for item in archive.infolist() if Path(item.filename).name == 'dcm2niix' and not item.is_dir()]
    if len(candidates) != 1 or candidates[0].file_size > 16 * 1024 * 1024:
        raise SystemExit('Unexpected converter archive; nothing installed.')
    binary = archive.read(candidates[0])
if hashlib.sha256(binary).hexdigest() != binary_sha:
    raise SystemExit('Converter executable failed its SHA-256 check; nothing installed.')
folder.mkdir(parents=True, exist_ok=True)
descriptor, name = tempfile.mkstemp(dir=folder, prefix='.dcm2niix-')
temporary = Path(name)
try:
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(binary)
    temporary.chmod(0o750)
    temporary.replace(target)
finally:
    temporary.unlink(missing_ok=True)
print('Installed verified dcm2niix ' + version + ': ' + str(target))
PY
