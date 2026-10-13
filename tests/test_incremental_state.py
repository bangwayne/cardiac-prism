"""Local saves and lazy priors must preserve scopes, undo and atomic recovery."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from server.domain import ProjectStore
from server.inference import padded_prior
from server.state import LazyPrior
import test_platform as fixtures


class IncrementalStateTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.PlatformTests()
        self.f.setUp()
        self.p = self.f.project

    def tearDown(self):
        self.f.tearDown()

    def reopen(self):
        p = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.f.app.state.store.projects[self.f.id] = self.f.project = self.p = p
        return p

    def all_priors(self):
        result = self.f.wait_job(self.f.post('segment-all-time-frames')['id'])
        self.assertEqual(result['state'], 'completed')
        self.p.save()

    def test_reopen_and_undo_snapshot_do_not_read_any_prior_and_click_reads_only_one(self):
        self.all_priors()
        expected = self.p.contexts[(1, 2, 2)]['prior'].copy()
        p = self.reopen()
        self.assertEqual(len(p.contexts), 18)
        self.assertTrue(all(isinstance(context['prior'], LazyPrior) for context in p.contexts.values()))
        p.snapshot(None, 'whole-study undo reference')
        self.assertEqual(p.prior_store.reads, 0)
        self.f.predictor.calls.clear()
        self.assertEqual(self.f.infer(label=2, t=1, z=2, mode='organ_click')['state'], 'completed')
        self.assertEqual(p.prior_store.reads, 1)
        np.testing.assert_array_equal(self.f.predictor.calls[-1]['prior'], expected)

    def test_single_layer_brush_save_copies_one_mask_and_prior_and_reads_no_others(self):
        self.all_priors()
        p = self.reopen()
        before = {label: mask.copy() for label, mask in p.masks.items()}
        self.f.brush(label=2, points=[[11, 8]], radius=0)
        p.save()
        self.assertEqual(p.prior_store.reads, 0)
        self.assertEqual(set(p.last_save['arrays']), {'mask_0_1_2', 'prior_0_1_2'})
        self.assertEqual(p.last_save['copied_bytes'], 12 * 16 + 512 * 512 * 4)
        after = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        before[2][0, 1, 8, 11] = 1
        for label in before:
            np.testing.assert_array_equal(after.masks[label], before[label])
        np.testing.assert_array_equal(after.contexts[(0, 1, 2)]['prior'], padded_prior(before[2][0, 1]))

    def test_collision_saves_only_two_affected_organs_on_the_same_layer(self):
        self.all_priors()
        p = self.reopen()
        self.f.brush(label=2, points=[[6, 5]], radius=0)
        p.save()
        self.assertEqual(p.prior_store.reads, 0)
        self.assertEqual(set(p.last_save['arrays']), {'mask_0_1_1', 'mask_0_1_2', 'prior_0_1_1', 'prior_0_1_2'})
        after = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.assertEqual(after.masks[1][0, 1, 5, 6], 0)
        self.assertEqual(after.masks[2][0, 1, 5, 6], 1)
        for label in [1, 2, 3]:
            np.testing.assert_array_equal(after.masks[label], p.masks[label])

    def test_saved_undo_redo_reuses_immutable_priors_and_retains_referenced_files(self):
        for x in [6, 7, 8]:
            self.f.brush(label=2, points=[[x, 5]], radius=0)
            self.p.save()
        self.f.post('undo')
        self.p.save()
        self.assertEqual(self.p.prior_store.reads, 0)
        self.assertEqual(self.p.last_save['arrays'], ['mask_0_1_2'])
        undo = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.assertEqual(undo.masks[2][0, 1, 5, 8], 0)
        np.testing.assert_array_equal(undo.contexts[(0, 1, 2)]['prior'], padded_prior(undo.masks[2][0, 1]))
        self.f.post('redo')
        self.p.save()
        redo = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.assertEqual(redo.masks[2][0, 1, 5, 8], 1)
        np.testing.assert_array_equal(redo.contexts[(0, 1, 2)]['prior'], padded_prior(redo.masks[2][0, 1]))

    def test_metadata_only_save_does_not_create_or_read_array_files(self):
        self.all_priors()
        p = self.reopen()
        self.f.post('lock', organ=2, locked=True)
        p.save()
        self.assertEqual(p.last_save['arrays'], [])
        self.assertEqual(p.last_save['written_bytes'], 0)
        self.assertEqual(p.prior_store.reads, 0)
        self.assertTrue(ProjectStore(self.f.root / 'work/projects').get(self.f.id).locks[2])

    def test_failed_manifest_publication_keeps_the_previous_complete_state(self):
        original = (self.p.directory / 'project.json').read_bytes()
        self.f.brush(label=2, points=[[6, 5]], radius=0)
        with patch('server.domain.atomic_json', side_effect=OSError('publication failed')):
            with self.assertRaises(OSError):
                self.p.save()
        self.assertEqual((self.p.directory / 'project.json').read_bytes(), original)
        old = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.assertFalse(old.masks[2].any())
        self.assertEqual(list(self.p.directory.glob('layer-*.npz')), [])
        self.p.save()
        self.assertEqual(ProjectStore(self.f.root / 'work/projects').get(self.f.id).masks[2][0, 1, 5, 6], 1)

    def test_new_layer_edited_during_save_remains_dirty_and_is_saved_next(self):
        self.f.brush(label=2, points=[[6, 5]], radius=0)
        revision = self.p.revision
        entered, release = threading.Event(), threading.Event()
        original = zipfile.ZipFile

        class SlowArchive(original):
            def __init__(self, *args, **kwargs):
                if kwargs.get('compression') == zipfile.ZIP_DEFLATED:
                    entered.set()
                    if not release.wait(6):
                        raise TimeoutError('Local edit was blocked')
                super().__init__(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=2) as executor, patch('server.state.zipfile.ZipFile', SlowArchive):
            saved = executor.submit(self.p.save)
            try:
                self.assertTrue(entered.wait(2))
                executor.submit(self.f.brush, label=3, t=1, z=2, points=[[11, 8]], radius=0).result(timeout=2)
            finally:
                release.set()
            saved.result(timeout=5)
        self.assertEqual(self.p.saved_revision, revision)
        self.assertEqual(self.p.dirty_layers, {(1, 2, 3): revision + 1})
        self.p.save()
        self.assertEqual(set(self.p.last_save['arrays']), {'mask_1_2_3', 'prior_1_2_3'})
        after = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        self.assertEqual(after.masks[2][0, 1, 5, 6], 1)
        self.assertEqual(after.masks[3][1, 2, 8, 11], 1)

    def test_schema_one_legacy_priors_stay_lazy_and_migrate_without_rewriting_base(self):
        path = self.p.directory / 'project.json'
        manifest = json.loads(path.read_text())
        manifest.update(schema_version=1, state_file='state-legacy.npz', contexts=[])
        for field in ['storage', 'mask_layers', 'reference_file']:
            manifest.pop(field, None)
        arrays = {f'mask_{label}': mask for label, mask in self.p.masks.items()}
        probability = np.full((1, 512, 512), .337, dtype=np.float32)
        for t in range(2):
            for z in range(3):
                for label in [1, 2, 3]:
                    name = f'prior_{t}_{z}_{label}'
                    arrays[name] = probability
                    manifest['contexts'].append(dict(key=[t, z, label], points=[], iteration=1, mode='organ', prior_key=name))
        base = self.p.directory / manifest['state_file']
        np.savez_compressed(base, **arrays)
        path.write_text(json.dumps(manifest))
        digest = hashlib.sha256(base.read_bytes()).hexdigest()
        p = self.reopen()
        self.assertEqual(p.prior_store.reads, 0)
        self.f.brush(label=2, points=[[11, 8]], radius=0)
        p.save()
        self.assertEqual(p.prior_store.reads, 0)
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), digest)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['schema_version'], 2)
        self.assertEqual(saved['state_file'], 'state-legacy.npz')
        again = ProjectStore(self.f.root / 'work/projects').get(self.f.id)
        for key, context in again.contexts.items():
            if key != (0, 1, 2):
                np.testing.assert_array_equal(context['prior'], probability)

    def test_slice_mask_endpoint_returns_only_the_requested_plane_with_revision(self):
        self.f.brush(label=2, t=1, z=2, points=[[6, 5]], radius=0)
        response = self.f.client.get(self.f.prefix + '/masks?time=1&slice=2')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Mask-Slice'], '2')
        self.assertEqual(int(response.headers['X-Project-Revision']), self.p.revision)
        self.assertEqual(len(response.content), 3 * 12 * 16)
        np.testing.assert_array_equal(np.frombuffer(response.content, dtype=np.uint8).reshape(3, 12, 16),
                                      np.stack([self.p.masks[label][1, 2] for label in [1, 2, 3]]))
        self.assertEqual(self.f.client.get(self.f.prefix + '/masks?time=1&slice=3').status_code, 400)


if __name__ == '__main__':
    unittest.main()
