"""Clear scopes, external baselines, lazy first-click priors and exact undo."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from server.domain import ProjectStore
from server.inference import padded_prior
import test_platform as fixtures


class RefinementTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job
    annotation = fixtures.PlatformTests.annotation

    def seed(self):
        for t in range(2):
            for z in range(3):
                for label in (1, 2, 3):
                    self.brush(label, t, z, points=[[label * 3, 3 + t]], radius=0)

    def assert_masks(self, expected):
        for label in (1, 2, 3):
            np.testing.assert_array_equal(self.project.masks[label], expected[label])

    def paired_fixture(self):
        images = self.root / 'demo-data/images'; images.mkdir(parents=True)
        (images / 'oblique4d.nii.gz').symlink_to(self.path)
        folder = self.root / 'demo-data/derivate/sax_itksnap_safe'; folder.mkdir(parents=True)
        labels = np.zeros(self.project.voxels.shape, dtype=np.uint8)
        labels[:, :, 2:4, 2:4] = 1
        labels[:, :, 5:7, 6:8] = 2
        labels[:, :, 8:10, 11:13] = 3
        annotation = sitk.GetImageFromArray(labels, isVector=False); annotation.CopyInformation(self.project.image)
        path = folder / 'oblique4d_cinema_rv1lv3.nii.gz'; sitk.WriteImage(annotation, str(path))
        entry = {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        registry = self.root / 'demo-data/derivate/refinement-masks.json'
        registry.write_text(json.dumps({'oblique4d.nii.gz': entry}))
        (self.root / 'demo-data/provenance.json').write_text('[]')
        return labels, path, registry

    def test_clear_all_current_slice_leaves_other_frames_layers_and_reference(self):
        self.seed(); before = {k: v.copy() for k, v in self.project.masks.items()}
        contexts = copy.deepcopy(self.project.contexts)
        self.project.reference = np.ones(self.project.voxels.shape, dtype=np.uint8)
        undo = len(self.project.undo_stack)
        self.post('clear-all-masks-slice', time=1, slice=1)
        for label in (1, 2, 3):
            expected = before[label].copy(); expected[1, 1] = 0
            np.testing.assert_array_equal(self.project.masks[label], expected)
            self.assertEqual(self.project.context((1, 1, label)), {'points': [], 'prior': None, 'iteration': 0, 'mode': None})
            np.testing.assert_array_equal(self.project.context((0, 1, label))['prior'], contexts[(0, 1, label)]['prior'])
        self.assertTrue(self.project.reference.all())
        self.assertEqual(len(self.project.undo_stack), undo + 1)
        self.post('undo'); self.assert_masks(before)
        for key in contexts:
            np.testing.assert_array_equal(self.project.contexts[key]['prior'], contexts[key]['prior'])

    def test_clear_layer_across_time_uses_small_snapshot_and_zero_prior_reads(self):
        self.seed(); self.project.save()
        before = {k: v.copy() for k, v in self.project.masks.items()}
        contexts = dict(self.project.contexts)
        self.post('clear-all-masks-layer', time=1, slice=1)
        snapshot = self.project.undo_stack[-1]
        self.assertEqual(snapshot.scope, (None, 1, (1, 2, 3)))
        self.assertEqual(sum(mask.nbytes for mask in snapshot.masks.values()), 3 * 2 * 12 * 16)
        self.assertEqual(self.project.prior_store.reads, 0)
        self.assertEqual(set(self.project.dirty_layers), {(t, 1, label) for t in range(2) for label in (1, 2, 3)})
        for label in (1, 2, 3):
            expected = before[label].copy(); expected[:, 1] = 0
            np.testing.assert_array_equal(self.project.masks[label], expected)
        for key, context in contexts.items():
            if key[1] != 1:
                self.assertIs(self.project.contexts[key], context)
        self.project.save(); self.assertEqual(self.project.prior_store.reads, 0)
        self.post('undo'); self.assert_masks(before)
        self.project.save(); self.assertEqual(self.project.prior_store.reads, 0)
        self.post('redo')
        self.assertTrue(all(not mask[:, 1].any() for mask in self.project.masks.values()))

    def test_all_mask_clear_rejects_locks_stale_revisions_and_bad_scope(self):
        self.seed(); before = {k: v.copy() for k, v in self.project.masks.items()}
        self.post('lock', organ=2, locked=True)
        revision = self.project.revision; undo = len(self.project.undo_stack)
        for action in ('clear-all-masks-slice', 'clear-all-masks-layer'):
            for request in ({'revision': revision, 'time': 0, 'slice': 1},
                            {'revision': revision - 1, 'time': 0, 'slice': 1},
                            {'revision': revision, 'time': 2, 'slice': 1},
                            {'revision': revision, 'time': 0, 'slice': 3}):
                result = self.client.post(self.prefix + '/' + action, json=request)
                self.assertEqual(result.status_code, 400, result.text)
        self.assert_masks(before); self.assertEqual(self.project.revision, revision)
        self.assertEqual(len(self.project.undo_stack), undo)

    def test_three_dimensional_layer_clear_is_the_current_slice(self):
        image = sitk.GetImageFromArray(np.ones((3, 12, 16), dtype=np.float32))
        path = self.root / 'three.nii.gz'; sitk.WriteImage(image, str(path))
        project = self.app.state.store.create(path, 'three.nii.gz')
        for label in (1, 2, 3):
            project.masks[label][0, :, label, label] = 1
        original = {k: v.copy() for k, v in project.masks.items()}
        response = self.client.post('/api/projects/' + project.id + '/clear-all-masks-layer',
                                    json={'revision': 0, 'time': 0, 'slice': 1})
        self.assertEqual(response.status_code, 200)
        for label in (1, 2, 3):
            expected = original[label].copy(); expected[0, 1] = 0
            np.testing.assert_array_equal(project.masks[label], expected)
        project.undo()
        for label in (1, 2, 3):
            np.testing.assert_array_equal(project.masks[label], original[label])

    def test_cleared_layer_save_reopen_has_no_resurrected_mask_or_prior(self):
        self.seed(); self.post('clear-all-masks-layer', time=0, slice=2); self.project.save()
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        for label in (1, 2, 3):
            np.testing.assert_array_equal(reopened.masks[label], self.project.masks[label])
            for t in range(2):
                self.assertIsNone(reopened.context((t, 2, label))['prior'])
        self.assertEqual(reopened.prior_store.reads, 0)
        self.infer(label=2, t=1, z=2)
        self.assertIsNone(self.predictor.calls[-1]['prior'])
        self.assertEqual(self.predictor.calls[-1]['iteration'], 0)

    def test_new_library_study_starts_with_exact_external_masks_without_ai(self):
        labels, _, _ = self.paired_fixture()
        with patch('server.app.ROOT', self.root):
            demos = self.client.get('/api/demos').json()
            self.assertTrue(demos[0]['has_annotation'])
            response = self.client.post('/api/demos/open', json={'name': 'oblique4d.nii.gz'})
            self.assertEqual(response.status_code, 200, response.text)
        project = self.app.state.store.get(response.json()['id'])
        self.assertEqual(project.undo_stack, [])
        self.assertEqual(self.predictor.calls, [])
        for label in (1, 2, 3):
            np.testing.assert_array_equal(project.masks[label], labels == label)
        self.assertTrue(all(context['mask_prior'] and context['prior'] is None for context in project.contexts.values()))
        saved = json.loads((project.directory / 'project.json').read_text())
        self.assertTrue(all('prior_key' not in record for record in saved['contexts']))
        self.assertEqual(project.prior_store.reads, 0)

    def test_existing_study_load_is_one_undo_and_keeps_reference(self):
        self.seed(); labels, _, _ = self.paired_fixture()
        before = {k: v.copy() for k, v in self.project.masks.items()}
        contexts = copy.deepcopy(self.project.contexts); undo = len(self.project.undo_stack)
        self.project.reference = np.ones(self.project.voxels.shape, dtype=np.uint8)
        with patch('server.app.ROOT', self.root):
            self.post('paired-annotation')
        self.assertEqual(len(self.project.undo_stack), undo + 1)
        for label in (1, 2, 3):
            np.testing.assert_array_equal(self.project.masks[label], labels == label)
        self.assertTrue(self.project.reference.all())
        self.post('undo'); self.assert_masks(before)
        for key in contexts:
            np.testing.assert_array_equal(self.project.contexts[key]['prior'], contexts[key]['prior'])
        self.post('redo')
        self.assertTrue(all(context.get('mask_prior') for context in self.project.contexts.values()))

    def test_first_click_uses_only_current_imported_mask_as_native_prior(self):
        labels, _, _ = self.paired_fixture()
        with patch('server.app.ROOT', self.root):
            self.post('paired-annotation')
        original = {k: v.copy() for k, v in self.project.masks.items()}
        result = self.infer(label=2, mode='organ_click', t=1, z=2)
        self.assertEqual(result['state'], 'completed', result)
        call = self.predictor.calls[-1]
        np.testing.assert_array_equal(call['prior'], padded_prior(labels[1, 2] == 2))
        self.assertEqual(call['iteration'], 1)
        self.assertNotIn('mask_prior', self.project.contexts[(1, 2, 2)])
        for label in (1, 3):
            np.testing.assert_array_equal(self.project.masks[label], original[label])
        np.testing.assert_array_equal(self.project.masks[2][0], original[2][0])
        np.testing.assert_array_equal(self.project.masks[2][1, :2], original[2][1, :2])
        self.post('undo'); self.assert_masks(original)
        self.assertTrue(self.project.contexts[(1, 2, 2)]['mask_prior'])

    def test_saved_external_baseline_keeps_lazy_first_click_behavior(self):
        labels, _, _ = self.paired_fixture()
        with patch('server.app.ROOT', self.root):
            self.post('paired-annotation')
        self.project.save()
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.prior_store.reads, 0)
        self.app.state.store.projects[self.id] = self.project = reopened
        self.infer(label=3, mode='click', t=0, z=0)
        np.testing.assert_array_equal(self.predictor.calls[-1]['prior'], padded_prior(labels[0, 0] == 3))
        self.assertEqual(self.predictor.calls[-1]['iteration'], 1)

    def test_paired_validation_failure_never_overwrites_existing_edits(self):
        self.seed(); _, path, registry = self.paired_fixture()
        before = {k: v.copy() for k, v in self.project.masks.items()}; revision = self.project.revision
        with patch('server.app.ROOT', self.root):
            path.write_bytes(path.read_bytes() + b'changed')
            response = self.client.post(self.prefix + '/paired-annotation', json={'revision': revision})
            self.assertEqual(response.status_code, 400)
            entry = {'filename': '../outside.nii.gz', 'sha256': 'x'}
            registry.write_text(json.dumps({'oblique4d.nii.gz': entry}))
            response = self.client.post(self.prefix + '/paired-annotation', json={'revision': revision})
            self.assertEqual(response.status_code, 400)
        self.assert_masks(before); self.assertEqual(self.project.revision, revision)

    def test_new_library_pair_geometry_failure_leaves_no_partial_study(self):
        _, path, registry = self.paired_fixture()
        mask = sitk.ReadImage(str(path)); mask.SetSpacing((1., 1., 1., .04)); sitk.WriteImage(mask, str(path))
        registry.write_text(json.dumps({'oblique4d.nii.gz': {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}))
        before = len(self.app.state.store.list())
        with patch('server.app.ROOT', self.root):
            response = self.client.post('/api/demos/open', json={'name': 'oblique4d.nii.gz'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.app.state.store.list()), before)


if __name__ == '__main__':
    unittest.main()
