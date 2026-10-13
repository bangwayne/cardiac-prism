"""Portable partial locks and migration from previously saved full-frame locks."""
import hashlib
import json
import unittest
import zipfile

import numpy as np

import test_platform as fixtures
from server.domain import ProjectStore
from server.study_backup import create_backup, import_backup


class SliceLockBackupTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job

    def archive(self):
        self.infer(label=2, t=0, z=1)
        self.brush(label=3, t=1, z=2, radius=0)
        self.post('slice-locks', times=[0], slices=[1], locked=True)
        return create_backup(self.project)

    def rewritten(self, source, change):
        with zipfile.ZipFile(source) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(files['project.json'])
        change(manifest)
        files['project.json'] = json.dumps(manifest).encode()
        inventory = json.loads(files['backup.json'])
        inventory['files']['project.json'] = hashlib.sha256(files['project.json']).hexdigest()
        files['backup.json'] = json.dumps(inventory).encode()
        path = self.root/'modified-slice-locks.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in files.items():
                archive.writestr(name, value)
        return path

    def test_partial_lock_roundtrip_protects_only_chosen_plane(self):
        source = self.archive()
        restored = import_backup(self.app.state.store, source)
        self.assertEqual(restored.locked_slices, {(0, 1)})
        self.assertEqual(restored.view()['locked_frames'], [])
        self.assertEqual(restored.view()['locked_slices'], [[0, 1]])
        for label in (1, 2, 3):
            np.testing.assert_array_equal(restored.masks[label], self.project.masks[label])
        for name in ('points', 'iteration', 'mode'):
            self.assertEqual(restored.contexts[(0, 1, 2)][name], self.project.contexts[(0, 1, 2)][name])
        np.testing.assert_array_equal(np.asarray(restored.contexts[(0, 1, 2)]['prior']),
                                      np.asarray(self.project.contexts[(0, 1, 2)]['prior']))
        prefix = '/api/projects/'+restored.id
        locked = self.client.post(prefix+'/brush', json={'revision': restored.revision, 'time': 0,
            'slice': 1, 'organ': 2, 'points': [[6, 5]], 'radius': 0})
        self.assertEqual(locked.status_code, 400)
        adjacent = self.client.post(prefix+'/brush', json={'revision': restored.revision, 'time': 0,
            'slice': 2, 'organ': 2, 'points': [[6, 5]], 'radius': 0})
        self.assertEqual(adjacent.status_code, 200, adjacent.text)
        restored.save()
        reopened = ProjectStore(restored.directory.parent).get(restored.id)
        self.assertEqual(reopened.locked_slices, {(0, 1)})

    def test_legacy_full_frame_backup_migrates_and_can_unlock_one_layer(self):
        source = self.archive()
        def legacy(manifest):
            manifest.pop('locked_slices', None)
            manifest['locked_frames'] = [0]
        restored = import_backup(self.app.state.store, self.rewritten(source, legacy))
        self.assertEqual(restored.locked_slices, {(0, 0), (0, 1), (0, 2)})
        response = self.client.post('/api/projects/'+restored.id+'/slice-locks', json={
            'revision': restored.revision, 'times': [0], 'slices': [1], 'locked': False})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['locked_frames'], [])
        self.assertEqual(restored.locked_slices, {(0, 0), (0, 2)})
        reopened = ProjectStore(restored.directory.parent).get(restored.id)
        self.assertEqual(reopened.locked_slices, {(0, 0), (0, 2)})

    def test_canonical_empty_locks_override_legacy_full_frame_metadata(self):
        source = self.archive()
        def canonical(manifest):
            manifest['locked_slices'] = []
            manifest['locked_frames'] = [0]
        restored = import_backup(self.app.state.store, self.rewritten(source, canonical))
        self.assertEqual(restored.locked_slices, set())
        self.assertEqual(restored.view()['locked_frames'], [])

    def test_invalid_plane_indices_rejected_before_study_publication(self):
        source = self.archive()
        directory = self.app.state.store.directory
        before = {path.name for path in directory.iterdir()}
        invalid = ([[True, 0]], [[0, False]], [[0, -1]], [[2, 0]], [[0, 3]],
                   [[0, 1], [0, 1]], [[0, '1']], [[0]], [0, 1], None, {'0': [1]})
        for value in invalid:
            with self.subTest(locked_slices=value):
                path = self.rewritten(source, lambda manifest: manifest.update(locked_slices=value))
                with self.assertRaises(ValueError):
                    import_backup(self.app.state.store, path)
                self.assertEqual({entry.name for entry in directory.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
