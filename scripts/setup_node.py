#!/usr/bin/env python3
"""Install a versioned Node.js runtime when Node.js 22 is not on PATH."""
import hashlib
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '22.15.0'


def extract_runtime(archive, directory):
    """Extract files safely, then create npm's relative executable links.

    Some Python 3.10 distribution backports resolve a symlink against the
    archive root instead of its containing directory when using filter='data'.
    Keeping links out of tarfile's extraction also makes that check explicit.
    """
    root = Path(directory).resolve()
    files, links = [], []
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or '..' in name.parts or not (root / name).resolve().is_relative_to(root):
            raise ValueError('Node.js archive contains an unsafe path.')
        if member.issym():
            links.append(member)
        elif member.isfile() or member.isdir():
            files.append(member)
        else:
            raise ValueError('Node.js archive contains an unsupported file type.')
    archive.extractall(directory, members=files, filter='data')
    for member in links:
        destination = root / member.name
        target = (destination.parent / member.linkname).resolve()
        if Path(member.linkname).is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Node.js archive contains an unsafe executable link.')
        destination.symlink_to(member.linkname)


def suitable(path):
    if not path:
        return False
    try:
        result = subprocess.run([str(path), '--version'], check=True,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip().startswith('v22.')
    except (OSError, subprocess.SubprocessError):
        return False


def main():
    target = ROOT / 'runtime/node'
    if suitable(target / 'bin/node') or suitable(shutil.which('node')):
        print('Node.js 22 is available.')
        return
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise SystemExit('Install Node.js 22 for this platform and run setup again.')
    name = f'node-v{VERSION}-linux-x64.tar.xz'
    base = f'https://nodejs.org/dist/v{VERSION}/'
    with urllib.request.urlopen(base + 'SHASUMS256.txt', timeout=60) as response:
        checksums = response.read(1024 * 1024).decode('ascii')
    expected = next((line.split()[0] for line in checksums.splitlines()
                     if len(line.split()) == 2 and line.split()[1] == name), None)
    if not expected:
        raise SystemExit('Node.js release checksum not found.')
    with urllib.request.urlopen(base + name, timeout=120) as response:
        payload = response.read(80 * 1024 * 1024 + 1)
    if len(payload) > 80 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit('Node.js archive failed its SHA-256 check.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.node-install-') as directory:
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:xz') as archive:
            extract_runtime(archive, directory)
        unpacked = Path(directory) / f'node-v{VERSION}-linux-x64'
        if not suitable(unpacked / 'bin/node'):
            raise SystemExit('Downloaded Node.js cannot run on this server.')
        if target.exists():
            raise SystemExit(f'An incompatible runtime exists at {target}; move it aside and retry.')
        os.replace(unpacked, target)
    print(f'Installed verified Node.js {VERSION}.')


if __name__ == '__main__':
    main()

