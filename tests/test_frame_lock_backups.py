"""Portable frame protection: restored studies preserve it, corrupt indices fail."""
import hashlib
import io
import json
import unittest
import zipfile

import numpy as np

import test_platform as fixtures
from server.domain import ProjectStore
from server.study_backup import create_backup, import_backup


class FrameLockBackupTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job

    def archive(self):
        self.infer(label=2, t=0, z=1)
        self.brush(label=3, t=1, z=2, radius=0)
        self.post('frame-locks', times=[0], locked=True)
        return create_backup(self.project)

    def rewritten_archive(self, source, locks, absent=False):
        """Rehash edited metadata to exercise validation beyond ZIP integrity."""
        with zipfile.ZipFile(source) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(files['project.json'])
        if absent:
            manifest.pop('locked_frames', None)
            manifest.pop('locked_slices', None)
        else:
            # Exercise the legacy representation independently of the canonical
            # slice-lock field that new archives also contain.
            manifest.pop('locked_slices', None)
            manifest['locked_frames'] = locks
        files['project.json'] = json.dumps(manifest).encode()
        inventory = json.loads(files['backup.json'])
        inventory['files']['project.json'] = hashlib.sha256(files['project.json']).hexdigest()
        files['backup.json'] = json.dumps(inventory).encode()
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        path = self.root / 'modified.zip'
        path.write_bytes(output.getvalue())
        return path

    def test_roundtrip_protection_masks_prompts_priors_and_input_profile(self):
        archive = self.archive()
        restored = import_backup(self.app.state.store, archive)
        self.assertNotEqual(restored.id, self.id)
        self.assertEqual(restored.locked_frames, {0})
        self.assertEqual(restored.input_profile, self.project.input_profile)
        for label in (1, 2, 3):
            np.testing.assert_array_equal(restored.masks[label], self.project.masks[label])
        original = self.project.contexts[(0, 1, 2)]
        copied = restored.contexts[(0, 1, 2)]
        self.assertEqual(copied['points'], original['points'])
        self.assertEqual(copied['iteration'], original['iteration'])
        np.testing.assert_array_equal(np.asarray(copied['prior']), np.asarray(original['prior']))
        revision = restored.revision
        response = self.client.post('/api/projects/' + restored.id + '/clear-slice',
                                    json={'revision': revision, 'time': 0, 'slice': 1, 'organ': 2})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(restored.revision, revision)
        self.assertEqual(restored.undo_stack, [])
        reopened = ProjectStore(restored.directory.parent).get(restored.id)
        self.assertEqual(reopened.locked_frames, {0})

    def test_legacy_archive_without_lock_metadata_opens_unlocked(self):
        archive = self.archive()
        old = self.rewritten_archive(archive, None, absent=True)
        restored = import_backup(self.app.state.store, old)
        self.assertEqual(restored.locked_frames, set())
        np.testing.assert_array_equal(restored.masks[2], self.project.masks[2])

    def test_invalid_protection_rejected_before_any_project_is_published(self):
        archive = self.archive()
        before = {path.name for path in self.app.state.store.directory.iterdir()}
        for invalid in ([True], [-1], [2], [0, 0], ['0'], 0, None, {'0': True}):
            with self.subTest(locks=invalid):
                changed = self.rewritten_archive(archive, invalid)
                with self.assertRaises(ValueError):
                    import_backup(self.app.state.store, changed)
                self.assertEqual({path.name for path in self.app.state.store.directory.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
