"""Saved annotations, portable backups and catalog removal must be lossless."""
import hashlib
import io
import json
import threading
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_platform as fixtures
from server.domain import ProjectStore
from server.study_notes import StudyNotesStore


class StudyLifecycleTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush

    def backup(self):
        response = self.client.get(self.prefix + '/export?kind=study')
        self.assertEqual(response.status_code, 200, response.text)
        return response.content

    def restore(self, data):
        return self.client.post('/api/projects/import-backup', files={'file': ('study.zip', data, 'application/zip')})

    def test_portable_backup_preserves_all_frames_contexts_geometry_reference_notes_and_locks(self):
        self.brush(t=0, z=1, label=1)
        self.brush(t=1, z=2, label=2)
        self.project.contexts[(1, 2, 2)]['points'] = [dict(x=6, y=5, positive=False)]
        self.project.reference = np.zeros(self.project.voxels.shape, np.uint8)
        self.project.reference[1, 2, 3:5, 4:6] = 3
        self.project.changed('import_reference')
        self.project.locks[3] = True
        note = dict(revision=0, text='全部时间帧已检查；可交接。', review_status='complete')
        self.assertEqual(self.client.post(self.prefix + '/notes', json=note).status_code, 200)
        priors = {key: value['prior'].copy() for key, value in self.project.contexts.items() if value['prior'] is not None}
        undo = len(self.project.undo_stack)
        data = self.backup()
        self.assertEqual(len(self.project.undo_stack), undo)
        self.assertEqual(self.project.revision, self.project.saved_revision)
        response = self.restore(data); self.assertEqual(response.status_code, 200, response.text)
        restored = self.app.state.store.get(response.json()['id'])
        self.assertNotEqual(restored.id, self.id)
        self.assertEqual(restored.source_path.read_bytes(), self.path.read_bytes())
        self.assertEqual(restored.source_geometry, self.project.source_geometry)
        self.assertEqual(restored.locks, self.project.locks)
        self.assertEqual(restored.revision, self.project.revision)
        self.assertEqual(restored.events, self.project.events)
        np.testing.assert_array_equal(restored.voxels, self.project.voxels)
        np.testing.assert_array_equal(restored.reference, self.project.reference)
        for label in self.project.masks:
            np.testing.assert_array_equal(restored.masks[label], self.project.masks[label])
        for key, context in self.project.contexts.items():
            self.assertEqual({k: v for k, v in context.items() if k != 'prior'},
                             {k: v for k, v in restored.contexts[key].items() if k != 'prior'})
            if key in priors:
                np.testing.assert_array_equal(restored.contexts[key]['prior'].copy(), priors[key])
        notes = StudyNotesStore(self.app.state.store.directory)
        self.assertEqual(notes.get(restored.id), notes.get(self.id))
        reopened = ProjectStore(self.app.state.store.directory).get(restored.id)
        for label in reopened.masks:
            np.testing.assert_array_equal(reopened.masks[label], restored.masks[label])
        self.assertEqual(reopened.undo_stack, [])

    def test_all_nifti_exports_import_voxel_exact_with_full_4d_geometry(self):
        self.brush(t=0, z=1, label=1)
        self.brush(t=1, z=2, label=2, points=[[12, 8]])
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        labelmap = self.client.get(self.prefix + '/export?kind=labelmap')
        self.assertEqual(labelmap.status_code, 200)
        for label in before:
            self.project.masks[label].fill(0)
        response = self.client.post(self.prefix + '/annotations', data=dict(revision=self.project.revision, target='annotation', binary_organ=0),
            files={'file': ('labels.nii.gz', labelmap.content)})
        self.assertEqual(response.status_code, 200, response.text)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])
        for label in [1, 2, 3]:
            data = self.client.get(self.prefix + '/export?kind=' + str(label)).content
            self.project.masks[label].fill(0)
            response = self.client.post(self.prefix + '/annotations', data=dict(revision=self.project.revision, target='annotation', binary_organ=label),
                files={'file': ('binary.nii.gz', data)})
            self.assertEqual(response.status_code, 200, response.text)
            for organ in before:
                np.testing.assert_array_equal(self.project.masks[organ], before[organ])

    def test_save_reopen_prior_is_lazy_and_notes_and_masks_are_exact(self):
        self.brush(t=1, z=2)
        before = self.project.contexts[(1, 2, 1)]['prior'].copy()
        self.client.post(self.prefix + '/notes', json=dict(revision=0, text='已完成', review_status='complete'))
        self.post('save')
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(reopened.prior_store.reads, 0)
        np.testing.assert_array_equal(reopened.contexts[(1, 2, 1)]['prior'].copy(), before)
        self.assertEqual(reopened.prior_store.reads, 1)
        np.testing.assert_array_equal(reopened.masks[1], self.project.masks[1])

    def test_remove_and_restore_example_persist_without_deleting_source_or_saved_study(self):
        examples = self.client.get('/api/demos').json()
        name = examples[0]['name']
        source = self.library_root / 'demo-data/images' / name
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        view = self.project.view()
        self.assertEqual(self.client.post('/api/demos/visibility', json=dict(name=name, hidden=True)).status_code, 200)
        self.assertNotIn(name, [entry['name'] for entry in self.client.get('/api/demos').json()])
        hidden = next(entry for entry in self.client.get('/api/demos?include_hidden=true').json() if entry['name'] == name)
        self.assertTrue(hidden['hidden'])
        from server.library_state import LibraryState
        self.assertIn(name, LibraryState(self.root / 'work').hidden())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual(self.project.view(), view)
        self.assertEqual(self.client.post('/api/demos/visibility', json=dict(name=name, hidden=False)).status_code, 200)
        self.assertIn(name, [entry['name'] for entry in self.client.get('/api/demos').json()])
        self.assertEqual(self.client.post('/api/demos/visibility', json=dict(name='../source.nii.gz', hidden=True)).status_code, 400)

    def test_invalid_backup_is_rejected_atomically_and_existing_studies_unchanged(self):
        baseline = self.project.view(); data = self.backup()
        baseline = self.project.view()
        for mode in ['traversal', 'tamper', 'plain_masks', 'duplicate']:
            buffer = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(data)) as original, zipfile.ZipFile(buffer, 'w') as edited:
                if mode == 'plain_masks':
                    edited.writestr('RV.nii.gz', b'x')
                else:
                    for name in original.namelist():
                        value = original.read(name)
                        if mode == 'tamper' and name == 'study-notes.json': value += b' '
                        edited.writestr(name, value)
                    if mode == 'traversal': edited.writestr('../bad.json', b'{}')
                    if mode == 'duplicate': edited.writestr('backup.json', b'{}')
            self.assertEqual(self.restore(buffer.getvalue()).status_code, 400, mode)
            self.assertEqual(self.project.view(), baseline)
            self.assertEqual(len(self.client.get('/api/projects').json()), 1)
            self.assertEqual(list(self.app.state.store.directory.glob('.import-*')), [])

    def test_backup_has_no_temporary_exports_and_can_run_without_prior_reads(self):
        self.brush(t=1, z=2); self.post('save')
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.app.state.store.projects[self.id] = reopened
        self.backup()
        self.assertEqual(reopened.prior_store.reads, 0)
        self.assertFalse(list(self.root.rglob('study-backup-*.zip')))

    def test_3d_input_and_backup_preserve_non512_size_and_physical_geometry(self):
        image = sitk.GetImageFromArray(np.arange(4 * 24 * 32, dtype=np.float32).reshape(4, 24, 32))
        image.SetSpacing((.9, 1.3, 7)); image.SetOrigin((14, -22, 33))
        image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
        path = self.root / 'three-dimensional.nii.gz'; sitk.WriteImage(image, str(path))
        project = self.app.state.store.create(path, path.name)
        project.masks[3][0, 2, 10:14, 12:16] = 1
        project.changed('manual_3d_fixture'); project.mark_dirty(None)
        data = self.client.get('/api/projects/' + project.id + '/export?kind=study').content
        response = self.restore(data); self.assertEqual(response.status_code, 200, response.text)
        restored = self.app.state.store.get(response.json()['id'])
        self.assertEqual(restored.image.GetDimension(), 3)
        self.assertEqual(restored.voxels.shape, (1, 4, 24, 32))
        self.assertEqual(restored.source_geometry, project.source_geometry)
        np.testing.assert_array_equal(restored.masks[3], project.masks[3])
        labels, _, _ = restored.export('labelmap')
        target = self.root / 'export3d.nii.gz'; target.write_bytes(labels)
        self.assertEqual(sitk.ReadImage(str(target)).GetDimension(), 3)


if __name__ == '__main__':
    unittest.main()
