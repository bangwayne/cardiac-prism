"""Saving must preserve one complete revision without blocking interactive edits."""
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from server.domain import ProjectStore
import test_platform as fixtures


class SnapshotSaveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PlatformTests()
        self.fixture.setUp()
        self.project = self.fixture.project

    def tearDown(self):
        self.fixture.tearDown()

    def test_slow_compression_does_not_block_views_or_new_clicks(self):
        f = self.fixture
        f.brush(label=2, points=[[6, 5]], radius=0)
        captured = self.project.snapshot(None, 'expected saved generation')
        revision = self.project.revision
        entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile

        class SlowArchive(original):
            def __init__(self, *args, **kwargs):
                if kwargs.get('compression') == zipfile.ZIP_DEFLATED:
                    entered.set()
                    if not release.wait(6):
                        raise TimeoutError('Save held up interactive requests')
                super().__init__(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=2) as executor, patch('server.domain.zipfile.ZipFile', SlowArchive):
            saved = executor.submit(f.client.post, f.prefix + '/save', json={'revision': revision})
            try:
                self.assertTrue(entered.wait(2))
                viewed = executor.submit(f.client.get, f.prefix)
                self.assertEqual(viewed.result(timeout=2).status_code, 200)
                job = f.infer(label=2, mode='organ_click', point={'x':6, 'y':5, 'positive':False})
                self.assertEqual(job['state'], 'completed')
                self.assertEqual(job['changed_voxels'], 1)
                self.assertFalse(job['point_foreground'])
                self.assertEqual(self.project.contexts[(0,1,2)]['points'][-1]['positive'], False)
                self.assertGreater(self.project.revision, revision)
            finally:
                release.set()
            self.assertEqual(saved.result(timeout=5).status_code, 200)
        self.assertEqual(self.project.saved_revision, revision)
        self.assertTrue(self.project.view()['dirty'])
        reopened = ProjectStore(f.root / 'work/projects').get(f.id)
        self.assertEqual(reopened.revision, revision)
        for label in [1,2,3]:
            np.testing.assert_array_equal(reopened.masks[label], captured.masks[label])
        np.testing.assert_array_equal(reopened.contexts[(0,1,2)]['prior'], captured.contexts[(0,1,2)]['prior'])
        self.project.save()
        newest = ProjectStore(f.root / 'work/projects').get(f.id)
        self.assertEqual(newest.revision, self.project.revision)
        np.testing.assert_array_equal(newest.masks[2], self.project.masks[2])

    def test_failed_compression_preserves_last_complete_generation(self):
        f = self.fixture
        original = (self.project.directory / 'project.json').read_bytes()
        f.brush(radius=0)
        with patch('server.domain.np.lib.format.write_array', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                self.project.save()
        self.assertEqual((self.project.directory / 'project.json').read_bytes(), original)
        self.assertEqual(len(list(self.project.directory.glob('state-*.npz'))), 1)
        self.assertTrue(self.project.view()['dirty'])

    def test_brush_and_erase_complete_while_saved_generation_is_compressing(self):
        f = self.fixture
        f.brush(label=2, points=[[6, 5]], radius=0)
        captured = self.project.snapshot(None, 'generation before manual refinement')
        revision, undo_count = self.project.revision, len(self.project.undo_stack)
        entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile

        class SlowArchive(original):
            def __init__(self, *args, **kwargs):
                if kwargs.get('compression') == zipfile.ZIP_DEFLATED:
                    entered.set()
                    if not release.wait(8):
                        raise TimeoutError('Manual refinement was blocked by compression')
                super().__init__(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=3) as executor, patch('server.domain.zipfile.ZipFile', SlowArchive):
            saved = executor.submit(f.client.post, f.prefix + '/save', json={'revision': revision})
            try:
                self.assertTrue(entered.wait(2))
                executor.submit(f.brush, label=2, points=[[9, 5]], radius=0).result(timeout=2)
                executor.submit(f.brush, label=2, points=[[6, 5]], radius=0, erase=True).result(timeout=2)
                self.assertEqual(self.project.masks[2][0, 1, 5, 9], 1)
                self.assertEqual(self.project.masks[2][0, 1, 5, 6], 0)
                self.assertEqual(self.project.revision, revision + 2)
                self.assertEqual(len(self.project.undo_stack), undo_count + 2)
                for label in [1, 3]:
                    np.testing.assert_array_equal(self.project.masks[label], captured.masks[label])
                np.testing.assert_array_equal(self.project.masks[2][1], captured.masks[2][1])
            finally:
                release.set()
            self.assertEqual(saved.result(timeout=5).status_code, 200)
        self.assertEqual(self.project.saved_revision, revision)
        self.assertTrue(self.project.view()['dirty'])
        reopened = ProjectStore(f.root / 'work/projects').get(f.id)
        for label in [1, 2, 3]:
            np.testing.assert_array_equal(reopened.masks[label], captured.masks[label])
        np.testing.assert_array_equal(reopened.contexts[(0, 1, 2)]['prior'], captured.contexts[(0, 1, 2)]['prior'])
        self.project.save()
        newest = ProjectStore(f.root / 'work/projects').get(f.id)
        np.testing.assert_array_equal(newest.masks[2], self.project.masks[2])
        np.testing.assert_array_equal(newest.contexts[(0, 1, 2)]['prior'], self.project.contexts[(0, 1, 2)]['prior'])

    def test_idempotent_save_keeps_generation_and_attached_provenance_is_persisted(self):
        path = self.project.directory / 'project.json'
        original = path.read_bytes()
        self.project.save()
        self.assertEqual(path.read_bytes(), original)
        self.project.provenance = {'source': 'test'}
        self.project.save()
        self.assertEqual(json.loads(path.read_text())['provenance'], self.project.provenance)

    def test_cached_frame_statistics_follow_edits_and_undo(self):
        f = self.fixture
        before = self.project.view(0)
        f.brush(label=3, t=1, radius=0)
        self.assertEqual(self.project.view(0)['structures'], before['structures'])
        self.assertEqual(self.project.view(1)['structures'][2]['voxel_count'], 1)
        f.post('undo')
        self.assertEqual(self.project.view(1)['structures'][2]['voxel_count'], 0)
