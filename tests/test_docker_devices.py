"""Explicit Docker bindings must respect GPU selection and kernel minor numbers."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('docker_devices', Path(__file__).resolve().parents[1] / 'scripts/docker_devices.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DockerDeviceTests(unittest.TestCase):
    def generate(self, selected, inventory='0, GPU-a\n1, GPU-b\n2, GPU-c\n'):
        detail = '<nvidia_smi_log>' + ''.join(
            f'<gpu><uuid>{uuid}</uuid><minor_number>{minor}</minor_number></gpu>'
            for uuid, minor in [('GPU-a', 3), ('GPU-b', 7), ('GPU-c', 1)]
        ) + '</nvidia_smi_log>'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(MODULE, 'ROOT', root), \
                    patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': selected}, clear=True), \
                    patch.object(MODULE.subprocess, 'run', side_effect=[SimpleNamespace(stdout=inventory), SimpleNamespace(stdout=detail)]), \
                    patch.object(Path, 'is_char_device', return_value=True):
                MODULE.main(explicit_bindings=True)
            return json.loads((root / 'runtime/compose.gpu-devices.json').read_text())['services']['workstation']

    def test_subset_indices_resolve_to_kernel_minors_and_complete_uuids(self):
        service = self.generate('1,0,1')
        self.assertEqual(service['environment']['NVIDIA_VISIBLE_DEVICES'], 'GPU-b,GPU-a')
        reservation = service['deploy']['resources']['reservations']['devices'][0]
        self.assertEqual(reservation['device_ids'], ['GPU-b', 'GPU-a'])
        self.assertNotIn('count', reservation)
        self.assertEqual(service['devices'][:2], ['/dev/nvidia7:/dev/nvidia7:rwm', '/dev/nvidia3:/dev/nvidia3:rwm'])
        self.assertNotIn('/dev/nvidia1:/dev/nvidia1:rwm', service['devices'])

    def test_uuid_selection_does_not_expose_unselected_devices(self):
        service = self.generate('GPU-c')
        self.assertEqual(service['environment']['NVIDIA_VISIBLE_DEVICES'], 'GPU-c')
        self.assertEqual(service['devices'], ['/dev/nvidia1:/dev/nvidia1:rwm',
                         '/dev/nvidiactl:/dev/nvidiactl:rwm', '/dev/nvidia-uvm:/dev/nvidia-uvm:rwm',
                         '/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools:rwm'])

    def test_unknown_or_empty_selection_never_falls_back_to_all(self):
        for selected in ('9', 'GPU-missing', '', 'none'):
            with self.subTest(selected=selected), self.assertRaises(SystemExit):
                self.generate(selected)

    def test_env_file_selection_is_used_without_executing_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.env').write_text('NVIDIA_VISIBLE_DEVICES="1,2"\n')
            with patch.object(MODULE, 'ROOT', root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(MODULE.selection(), '1,2')

    def test_standard_gpu_selection_uses_device_ids_without_exposing_all(self):
        for selected, expected in [('1,2,1', ['1', '2']), ('GPU-abcd, GPU-efgh', ['GPU-abcd', 'GPU-efgh'])]:
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(MODULE, 'ROOT', root), \
                        patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': selected}, clear=True), \
                        patch.object(MODULE.subprocess, 'run') as command:
                    MODULE.main()
                    command.assert_not_called()
                service = json.loads((root / 'runtime/compose.gpu-devices.json').read_text())['services']['workstation']
                reservation = service['deploy']['resources']['reservations']['devices'][0]
                self.assertEqual(reservation['device_ids'], expected)
                self.assertNotIn('count', reservation)
                self.assertNotIn('devices', service)

    def test_standard_all_selection_does_not_need_host_driver_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(MODULE, 'ROOT', root), patch.dict(os.environ, {}, clear=True), \
                    patch.object(MODULE.subprocess, 'run') as command:
                MODULE.main()
                command.assert_not_called()
            service = json.loads((root / 'runtime/compose.gpu-devices.json').read_text())['services']['workstation']
            self.assertEqual(service['deploy']['resources']['reservations']['devices'][0]['count'], 'all')
            self.assertNotIn('devices', service)

    def test_standard_empty_or_invalid_selection_is_rejected(self):
        for selected in ('', '1,', 'none', 'all,1', '/dev/nvidia0'):
            with self.subTest(selected=selected), patch.dict(os.environ, {'NVIDIA_VISIBLE_DEVICES': selected}, clear=True):
                with self.assertRaises(SystemExit):
                    MODULE.main()


if __name__ == '__main__':
    unittest.main()
