#!/usr/bin/env python3
"""Write a Compose GPU selection, optionally with explicit device bindings."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


def selection():
    value = os.environ.get('NVIDIA_VISIBLE_DEVICES')
    env_file = ROOT / '.env'
    if value is None and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, candidate = line.strip().removeprefix('export ').partition('=')
            if separator and key.strip() == 'NVIDIA_VISIBLE_DEVICES':
                value = candidate.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in '\"\'':
                    value = value[1:-1]
                else:
                    value = value.split('#', 1)[0].strip()
    return 'all' if value is None else value.strip()


def explicit_devices(requested):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    detail = subprocess.run(['nvidia-smi', '-q', '-x'], check=True,
                            capture_output=True, text=True, timeout=10)
    minors = {gpu.findtext('uuid'): gpu.findtext('minor_number')
              for gpu in ElementTree.fromstring(detail.stdout).findall('gpu')}
    devices = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 2 or not row[0].strip().isdigit():
            raise SystemExit('The NVIDIA driver returned an invalid GPU inventory.')
        uuid = row[1].strip()
        minor = minors.get(uuid)
        if minor is None or not minor.isdigit():
            raise SystemExit('The NVIDIA driver did not report a kernel device minor for a GPU.')
        devices[row[0].strip()] = {'uuid': uuid, 'minor': minor}
    indices = []
    for token in devices if requested == 'all' else requested.split(','):
        token = token.strip()
        index = token if token in devices else next((key for key, item in devices.items() if item['uuid'] == token), None)
        if index is None:
            raise SystemExit(f'Unknown GPU selection: {token}. Use physical GPU indices or complete GPU UUIDs.')
        if index not in indices:
            indices.append(index)
    if not indices:
        raise SystemExit('No GPUs selected. Use CPU mode or set NVIDIA_VISIBLE_DEVICES.')
    paths = [Path('/dev/nvidia' + devices[index]['minor']) for index in indices]
    paths += [Path('/dev/nvidiactl'), Path('/dev/nvidia-uvm')]
    if Path('/dev/nvidia-uvm-tools').is_char_device():
        paths.append(Path('/dev/nvidia-uvm-tools'))
    for path in paths:
        if not path.is_char_device():
            raise SystemExit(f'GPU device {path} is unavailable. Check the host NVIDIA driver before starting.')
    uuids = [devices[index]['uuid'] for index in indices]
    return uuids, paths


def main(explicit_bindings=None):
    requested = selection()
    if not requested:
        raise SystemExit('No GPUs selected. Use CPU mode or set NVIDIA_VISIBLE_DEVICES.')
    if explicit_bindings is None:
        explicit_bindings = os.environ.get('CARDIAC_PRISM_DOCKER_DEVICE_BINDINGS') == '1'
    paths = []
    reservation = {'driver': 'nvidia', 'capabilities': ['gpu']}
    if explicit_bindings:
        identifiers, paths = explicit_devices(requested)
        reservation['device_ids'] = identifiers
    elif requested == 'all':
        identifiers = ['all']
        reservation['count'] = 'all'
    else:
        identifiers = list(dict.fromkeys(token.strip() for token in requested.split(',')))
        if not all(re.fullmatch(r'[0-9]+|GPU-[A-Za-z0-9-]+', token) for token in identifiers):
            raise SystemExit('Use comma-separated physical GPU indices or complete GPU UUIDs.')
        reservation['device_ids'] = identifiers
    service = {
        'image': 'cardiac-prism:0.1.1-cuda11.7',
        'build': {'args': {'TORCH_VARIANT': 'cu117'}},
        'environment': {'CARDIAC_PRISM_DEVICE': '${CARDIAC_PRISM_DEVICE:-${KPRISM_DEVICE:-cuda:0}}',
                        'NVIDIA_VISIBLE_DEVICES': ','.join(identifiers),
                        'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'},
        'deploy': {'resources': {'reservations': {'devices': [
            reservation
        ]}}}
    }
    if paths:
        service['devices'] = [f'{path}:{path}:rwm' for path in paths]
    config = {'services': {'workstation': service}}
    target = ROOT / 'runtime/compose.gpu-devices.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(config, indent=2) + '\n')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(('Explicit GPU device bindings: ' if explicit_bindings else 'GPU selection: ') + ', '.join(identifiers))


if __name__ == '__main__':
    main()

