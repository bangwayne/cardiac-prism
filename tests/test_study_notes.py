"""Notes must persist without touching masks, priors, inference or undo history."""
import hashlib
import io
import json
import threading
import unittest
import zipfile
from unittest.mock import patch

import test_platform as _platform
from server.domain import ProjectStore


class StudyNotesTests(unittest.TestCase):
    setUp = _platform.PlatformTests.setUp
    tearDown = _platform.PlatformTests.tearDown
    post = _platform.PlatformTests.post
    brush = _platform.PlatformTests.brush

    def note(self, text='已标完 RV / MYO / LV。检查基底层。', status='complete', revision=0):
        return self.client.post(self.prefix + '/notes', json=dict(revision=revision, text=text, review_status=status))

    def files(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.project.directory.iterdir()
                if p.name != 'study-notes.json'}

    def test_notes_default_and_persistence_without_loading_image(self):
        default = self.client.get(self.prefix + '/notes').json()
        self.assertEqual(default, dict(revision=0, text='', review_status='in_progress', updated_at=None))
        before = self.files()
        del self.app.state.store.projects[self.id]
        response = self.note()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn(self.id, self.app.state.store.projects)
        self.assertEqual(before, self.files())
        record = self.client.get(self.prefix + '/notes').json()
        self.assertEqual(record['text'], '已标完 RV / MYO / LV。检查基底层。')
        self.assertEqual(record['revision'], 1)
        from server.study_notes import StudyNotesStore
        reopened = StudyNotesStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(reopened, record)

    def test_notes_preserve_masks_priors_revision_and_undo(self):
        self.brush()
        self.project.save()
        before = self.client.get(self.prefix).json()
        files = self.files()
        diagnostics = self.project.prior_store.diagnostics()
        self.assertEqual(self.note().status_code, 200)
        self.assertEqual(self.client.get(self.prefix).json(), before)
        self.assertEqual(self.files(), files)
        self.assertEqual(self.project.prior_store.diagnostics(), diagnostics)
        self.post('undo')
        self.assertEqual(self.client.get(self.prefix + '/notes').json()['review_status'], 'complete')

    def test_mask_save_and_reopen_keep_notes_and_library_summary(self):
        self.assertEqual(self.note().status_code, 200)
        self.brush(t=1, z=2)
        self.project.save()
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(reopened.revision, self.project.revision)
        record = self.client.get('/api/projects').json()[0]
        self.assertEqual(record['review_status'], 'complete')
        self.assertIn('已标完', record['note_preview'])
        self.assertIsNotNone(record['notes_updated_at'])

    def test_stale_note_and_invalid_payload_do_not_overwrite(self):
        self.assertEqual(self.note().status_code, 200)
        saved = self.client.get(self.prefix + '/notes').json()
        self.assertEqual(self.note('stale').status_code, 400)
        self.assertEqual(self.note('x' * 5001, revision=1).status_code, 422)
        self.assertEqual(self.note(status='unknown', revision=1).status_code, 422)
        self.assertEqual(self.client.get(self.prefix + '/notes').json(), saved)

    def test_clearing_note_and_completion_are_independent(self):
        self.assertEqual(self.note().status_code, 200)
        self.assertEqual(self.note('', 'in_progress', 1).status_code, 200)
        current = self.client.get(self.prefix + '/notes').json()
        self.assertEqual(current['text'], '')
        self.assertEqual(current['review_status'], 'in_progress')
        self.assertEqual(current['revision'], 2)
        self.assertEqual(self.note('', 'in_progress', 2).json(), current)

    def test_notes_work_during_inference_and_with_locked_organs(self):
        self.project.pending_job = 'running-job'
        self.project.locks = {1: True, 2: True, 3: True}
        self.assertEqual(self.note().status_code, 200)
        self.assertEqual(self.project.pending_job, 'running-job')
        self.assertEqual(self.project.revision, 0)

    def test_notes_remain_responsive_during_slow_array_save(self):
        self.brush()
        entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile

        def slow_archive(*args, **kwargs):
            if len(args) > 1 and args[1] == 'w':
                entered.set()
                self.assertTrue(release.wait(10))
            return original(*args, **kwargs)

        with patch('server.state.zipfile.ZipFile', slow_archive):
            worker = threading.Thread(target=self.project.save)
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.note().status_code, 200)
                self.assertTrue(worker.is_alive())
            finally:
                release.set(); worker.join(10)
        self.assertEqual(self.client.get(self.prefix + '/notes').json()['revision'], 1)

    def test_export_includes_notes_without_altering_labelmaps(self):
        self.assertEqual(self.note().status_code, 200)
        response = self.client.get(self.prefix + '/export?kind=bundle')
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            annotation = json.loads(archive.read('annotation.json'))
            self.assertEqual(annotation['study_notes']['review_status'], 'complete')
            self.assertIn('已标完', annotation['study_notes']['text'])


if __name__ == '__main__':
    unittest.main()
