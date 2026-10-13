"""Library actions must preserve unrelated studies and reject stale requests."""
import hashlib
import json
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import test_platform as fixtures
from server.domain import ProjectStore
from server.library_state import LibraryState


class LibraryManagementTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush

    def test_library_without_downloaded_catalog_still_opens_user_examples(self):
        (self.library_root / 'demo-data/provenance.json').unlink()
        entries = self.client.get('/api/demos').json()
        self.assertEqual(len(entries), 2)
        response = self.client.post('/api/demos/open', json={'name': entries[0]['name']})
        self.assertEqual(response.status_code, 200, response.text)

    def test_cine_catalog_metadata_does_not_depend_on_dicom_or_patient_filenames(self):
        record = {'name': 'example_4d.nii.gz', 'display_name': 'ACDC cine example',
                  'phase': '4D cine', 'time_frames': 2, 'slices': 3}
        (self.library_root / 'demo-data/provenance.json').write_text(json.dumps([record]))
        entry = next(item for item in self.client.get('/api/demos').json() if item['name'] == record['name'])
        self.assertEqual(entry['patient'], record['display_name'])
        self.assertEqual(entry['phase'], '4D cine')
        self.assertEqual((entry['time_frames'], entry['slices']), (2, 3))

    def manage(self, action, **values):
        note = self.client.get(self.prefix + '/notes').json()
        return self.client.post(self.prefix + '/' + action,
                                json={'revision': self.project.revision, 'notes_revision': note['revision'], **values})

    def files(self, directory=None):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (directory or self.project.directory).iterdir()}

    def test_complete_saves_dirty_masks_and_preserves_latest_note_prior_locks_and_undo(self):
        self.brush(t=1, z=2)
        self.client.post(self.prefix + '/notes', json=dict(revision=0, text='精修完成：保留备注。', review_status='in_progress'))
        self.project.locks[3] = True
        self.project.changed('lock', organ=3); self.project.mark_dirty(None)
        masks = {key: mask.copy() for key, mask in self.project.masks.items()}
        undo = len(self.project.undo_stack)
        result = self.manage('completion', review_status='complete')
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['text'], '精修完成：保留备注。')
        self.assertEqual(result.json()['review_status'], 'complete')
        self.assertEqual(self.project.saved_revision, self.project.revision)
        self.assertEqual(self.project.prior_store.reads, 0)
        self.assertEqual(len(self.project.undo_stack), undo)
        restored = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertTrue(restored.locks[3])
        for label in masks: np.testing.assert_array_equal(restored.masks[label], masks[label])
        self.assertEqual(self.client.get('/api/projects').json()[0]['review_status'], 'complete')

    def test_reopen_updates_completion_only_and_preserves_masks_notes_and_undo(self):
        self.brush(); self.manage('completion', review_status='complete')
        before = self.files(); view = self.project.view()
        result = self.manage('completion', review_status='in_progress')
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['review_status'], 'in_progress')
        self.assertEqual(self.project.view(), view)
        self.assertEqual({k: v for k, v in self.files().items() if k != 'study-notes.json'},
                         {k: v for k, v in before.items() if k != 'study-notes.json'})

    def test_completion_rejects_active_jobs_and_stale_note_or_mask_revisions(self):
        before = self.files()
        self.project.pending_job = 'active'
        self.assertEqual(self.manage('completion', review_status='complete').status_code, 400)
        self.project.pending_job = None
        self.assertEqual(self.manage('completion', review_status='complete', revision=1).status_code, 400)
        self.assertEqual(self.manage('completion', review_status='complete', notes_revision=1).status_code, 400)
        self.assertEqual(self.files(), before)
        self.assertEqual(self.client.get(self.prefix + '/notes').json()['review_status'], 'in_progress')

    def test_edit_during_completion_save_keeps_new_mask_and_does_not_mark_complete(self):
        self.brush(); entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile; results = []
        def slow(*args, **kwargs):
            if len(args) > 1 and args[1] == 'w':
                entered.set(); self.assertTrue(release.wait(10))
            return original(*args, **kwargs)
        with patch('server.state.zipfile.ZipFile', slow):
            worker = threading.Thread(target=lambda: results.append(self.manage('completion', review_status='complete')))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.brush(t=1, z=2, points=[[10, 8]])
            finally: release.set(); worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0].status_code, 400)
        self.assertTrue(self.project.masks[1][1, 2, 8, 10])
        self.assertNotEqual(self.project.saved_revision, self.project.revision)
        self.assertEqual(self.client.get(self.prefix + '/notes').json()['review_status'], 'in_progress')

    def test_delete_removes_only_requested_study_and_never_recreates_it(self):
        self.brush(); self.manage('completion', review_status='complete')
        other = self.app.state.store.create(self.path, self.path.name)
        original = self.path.read_bytes(); other_files = self.files(other.directory)
        result = self.manage('delete'); self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(self.project.directory.exists()); self.assertTrue(self.project.deleted)
        self.assertNotIn(self.id, self.app.state.store.projects)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.files(other.directory), other_files)
        self.assertEqual([item['id'] for item in self.client.get('/api/projects').json()], [other.id])
        for suffix in ['', '/notes', '/masks', '/volume']:
            self.assertEqual(self.client.get(self.prefix + suffix).status_code, 404, suffix)
        with self.assertRaises(KeyError): self.project.save()
        with self.assertRaises(KeyError): self.project.check_revision(self.project.revision)
        with self.assertRaises(KeyError): ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(list((self.root / 'work/.study-deletions').iterdir()), [])

    def test_delete_rejects_active_jobs_and_stale_mask_or_note_revisions(self):
        before = self.files(); self.project.pending_job = 'active'
        self.assertEqual(self.manage('delete').status_code, 400)
        self.project.pending_job = None
        self.assertEqual(self.manage('delete', revision=1).status_code, 400)
        self.assertEqual(self.manage('delete', notes_revision=1).status_code, 400)
        self.assertEqual(self.files(), before)
        self.assertEqual(self.client.get(self.prefix).status_code, 200)

    def test_delete_waits_for_inflight_save_and_then_cleans_the_entire_study(self):
        self.brush(); entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile; results = []
        def slow(*args, **kwargs):
            if len(args) > 1 and args[1] == 'w':
                entered.set(); self.assertTrue(release.wait(10))
            return original(*args, **kwargs)
        with patch('server.state.zipfile.ZipFile', slow):
            saver = threading.Thread(target=self.project.save); saver.start()
            self.assertTrue(entered.wait(5))
            deletion = threading.Thread(target=lambda: results.append(self.manage('delete'))); deletion.start()
            try:
                time.sleep(.1); self.assertTrue(deletion.is_alive()); self.assertTrue(self.project.directory.exists())
            finally: release.set(); saver.join(10); deletion.join(10)
        self.assertFalse(saver.is_alive()); self.assertFalse(deletion.is_alive())
        self.assertEqual(results[0].status_code, 200, results[0].text)
        self.assertFalse(self.project.directory.exists())

    def test_permanent_example_removal_excludes_both_lists_and_retains_original_and_saved_studies(self):
        names = [item['name'] for item in self.client.get('/api/demos').json()]
        name, other = names[:2]; source = self.library_root / 'demo-data/images' / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest(); files = self.files()
        self.assertEqual(self.client.post('/api/demos/delete', json={'name': name}).status_code, 400)
        self.assertEqual(self.client.post('/api/demos/visibility', json={'name': name, 'hidden': True}).status_code, 200)
        self.assertEqual(self.client.post('/api/demos/delete', json={'name': name}).status_code, 200)
        state = LibraryState(self.root / 'work')
        self.assertIn(name, state.deleted()); self.assertNotIn(name, state.hidden())
        self.assertNotIn(name, [item['name'] for item in self.client.get('/api/demos?include_hidden=true').json()])
        self.assertEqual(self.client.post('/api/demos/visibility', json={'name': name, 'hidden': False}).status_code, 400)
        self.assertEqual(self.client.post('/api/demos/open', json={'name': name}).status_code, 400)
        self.client.post('/api/demos/visibility', json={'name': other, 'hidden': True})
        self.client.post('/api/demos/visibility', json={'name': other, 'hidden': False})
        self.assertIn(name, state.deleted())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.files(), files)

    def test_invalid_ids_and_paths_cannot_delete_other_files(self):
        files = self.files()
        for project_id in ['not-a-study', 'f' * 32]:
            result = self.client.post('/api/projects/' + project_id + '/delete', json={'revision': 0, 'notes_revision': 0})
            self.assertEqual(result.status_code, 404)
        self.assertEqual(self.client.post('/api/demos/delete', json={'name': '../source.nii.gz'}).status_code, 400)
        self.assertEqual(self.files(), files)


if __name__ == '__main__':
    unittest.main()
