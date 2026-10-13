#!/usr/bin/env python3
"""Verify the inference dependencies before starting the workstation."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    args = parser.parse_args()
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append('Python 3.10 is required.')
    expected = {'torch': '1.13.1', 'torchvision': '0.14.1', 'numpy': '1.26.4',
                'SimpleITK': '2.3.1', 'opencv-python-headless': '4.9.0.80',
                'detectron2': '0.6', 'fastapi': '0.115.12', 'pydicom': '3.0.1'}
    print(f'Python {platform.python_version()} on {platform.system()} {platform.machine()}')
    for name, wanted in expected.items():
        try:
            actual = importlib.metadata.version(name)
            print(f'{name}: {actual}')
            if actual.split('+')[0] != wanted:
                errors.append(f'{name}: expected {wanted}, found {actual}.')
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'{name} is not installed.')
    for module in ('torchvision', 'SimpleITK', 'cv2', 'detectron2.layers',
                   'cardiac_prism.inference.inference', 'server.cardiac_model'):
        try:
            importlib.import_module(module)
        except Exception as error:
            errors.append(f'{module}: {error}')
    try:
        import torch
        from server.cuda_devices import device_properties
        available = torch.cuda.is_available()
        devices = device_properties()
        print(f'CUDA available: {available}; devices: {len(devices)}')
        if args.device == 'cuda' and not available:
            errors.append('CUDA was requested but is unavailable. Check the NVIDIA driver and GPU visibility.')
        if args.device == 'cuda' and available:
            (torch.ones((8, 8), device='cuda') @ torch.ones((8, 8), device='cuda')).cpu()
        for index, properties in enumerate(devices):
            print(f'GPU {index}: {properties.name}')
    except Exception as error:
        errors.append(f'PyTorch execution: {error}')
    try:
        from server.dicom_import import converter_path
        print(f'DICOM converter: {converter_path().name}')
    except Exception as error:
        errors.append(str(error))
    if errors:
        print('\nEnvironment check failed:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 1
    print('Environment check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

