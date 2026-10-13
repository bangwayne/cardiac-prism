"""Existing server settings remain usable after the project rename."""
import json
import os
from pathlib import Path
import subprocess
import unittest

from server.environment import apply_legacy_environment


class EnvironmentCompatibilityTests(unittest.TestCase):
    def test_python_and_shell_apply_identical_precedence(self):
        cases = [
            {'KPRISM_PORT': '8512', 'KPRISM_WORKSPACE': '/data/my studies',
             'VERSE_DEVICE': 'cpu', 'VERSE_CHECKPOINT': '/data/old.pth'},
            {'CARDIAC_PRISM_DEVICE': 'cpu', 'KPRISM_DEVICE': 'cuda:1', 'VERSE_DEVICE': 'cuda:2',
             'CARDIAC_PRISM_PORT': '8507', 'KPRISM_PORT': '8512'},
            {'KPRISM_DEVICE': 'cuda:1', 'VERSE_DEVICE': 'cuda:2', 'VERSE_CONFIG': '/data/config.yaml'},
            {},
        ]
        root = Path(__file__).resolve().parents[1]
        for case in cases:
            with self.subTest(case=case):
                translated = dict(case)
                apply_legacy_environment(translated)
                expected = {k: v for k, v in translated.items() if k.startswith('CARDIAC_PRISM_')}
                env = {k: v for k, v in os.environ.items()
                       if not k.startswith(('CARDIAC_PRISM_', 'KPRISM_', 'VERSE_'))}
                env.update(case)
                script = ('set -eu; source scripts/environment.sh; '
                          'python3 -c \'import json,os; print(json.dumps({k:v for k,v in os.environ.items() '
                          'if k.startswith("CARDIAC_PRISM_")}))\'')
                result = subprocess.run(['bash', '-c', script], cwd=root, env=env,
                                        check=True, text=True, capture_output=True)
                self.assertEqual(json.loads(result.stdout), expected)
        self.assertEqual(cases[0]['VERSE_DEVICE'], 'cpu')

    def test_legacy_alias_does_not_replace_current_configuration(self):
        settings = {'CARDIAC_PRISM_DEVICE': 'cpu', 'KPRISM_DEVICE': 'cuda:1', 'VERSE_DEVICE': 'cuda:2'}
        apply_legacy_environment(settings)
        self.assertEqual(settings['CARDIAC_PRISM_DEVICE'], 'cpu')
        settings = {'KPRISM_DEVICE': 'cuda:1', 'VERSE_DEVICE': 'cuda:2'}
        apply_legacy_environment(settings)
        self.assertEqual(settings['CARDIAC_PRISM_DEVICE'], 'cuda:1')
