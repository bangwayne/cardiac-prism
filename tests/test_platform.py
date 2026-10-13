"""State/geometry/API regression tests; fake inference is used only in tests."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from fastapi.testclient import TestClient

from server.app import create_app
from server.domain import ProjectStore, require_same_geometry
from server.inference import ModelService, padded_prior


class Predictor:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def __call__(self, image, points, label, mode, prior, iteration):
        self.calls.append({'points': [dict(p) for p in points], 'label': label, 'mode': mode,
                           'prior': None if prior is None else prior.copy(), 'iteration': iteration,
                           'image': image.copy()})
        self.entered.set()
        if self.block and not self.release.wait(10):
            raise RuntimeError('Test barrier timed out')
        if self.fail:
            raise RuntimeError('Deliberate test prediction failure')
        mask = np.zeros(image.shape, dtype=np.uint8)
        if not points:
            mask[3:7, 4:9] = 1
        for point in points:
            y, x = int(point['y']), int(point['x'])
            mask[max(0, y-1):y+2, max(0, x-1):x+2] = int(point['positive'])
        return {'mask': mask, 'probability': np.where(mask, .9, .1).astype(np.float32),
                'prior': padded_prior(mask), 'iteration': iteration + 1}


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        values = np.arange(2*3*12*16, dtype=np.float32).reshape(2, 3, 12, 16)
        image = sitk.GetImageFromArray(values, isVector=False)
        angle = np.pi / 6
        matrix = np.eye(4)
        matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        image.SetSpacing((.7, 1.2, 5., .04))
        image.SetOrigin((12., -25., 33., 0.))
        image.SetDirection(matrix.reshape(-1))
        self.path = self.root / 'oblique4d.nii.gz'
        sitk.WriteImage(image, str(self.path))
        self.library_root = self.root / 'sample-library'
        images = self.library_root / 'demo-data/images'
        images.mkdir(parents=True)
        for name in ('example_4d.nii.gz', 'second_4d.nii.gz'):
            (images / name).write_bytes(self.path.read_bytes())
        (self.library_root / 'demo-data/provenance.json').write_text('[]')
        self.root_patch = patch('server.app.ROOT', self.library_root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.predictor = Predictor()
        self.app = create_app(self.root / 'work', ModelService(self.predictor))
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        with self.path.open('rb') as stream:
            response = self.client.post('/api/projects', files={'file': ('oblique4d.nii.gz', stream)})
        self.assertEqual(response.status_code, 200, response.text)
        self.id = response.json()['id']
        self.project = self.app.state.store.get(self.id)
        self.prefix = '/api/projects/' + self.id

    def tearDown(self):
        self.predictor.release.set()
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def post(self, action, **details):
        response = self.client.post(self.prefix + '/' + action,
                                    json={'revision': self.project.revision, **details})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def brush(self, label=1, t=0, z=1, **details):
        return self.post('brush', time=t, slice=z, organ=label,
                         points=details.pop('points', [[6, 5]]), radius=details.pop('radius', 1), **details)

    def infer(self, label=1, mode='click', t=0, z=1, point=None, wait=True):
        details = dict(time=t, slice=z, organ=label, mode=mode)
        if mode != 'organ':
            details['point'] = point or {'x': 6., 'y': 5., 'positive': True}
        job = self.post('inference', **details)
        return self.wait_job(job['id']) if wait else job

    def wait_job(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = self.client.get('/api/jobs/' + job_id).json()
            if result['state'] not in ('running', 'queued'):
                return result
            time.sleep(.005)
        self.fail('Job did not finish')

    def annotation(self, image, target='annotation', binary=0):
        path = self.root / 'annotation.nii.gz'
        sitk.WriteImage(image, str(path))
        with path.open('rb') as stream:
            return self.client.post(self.prefix + '/annotations',
                                    data={'revision': self.project.revision, 'target': target, 'binary_organ': binary},
                                    files={'file': ('annotation.nii.gz', stream)})

    def test_four_dimensional_geometry_and_buffers(self):
        view = self.client.get(self.prefix).json()
        self.assertEqual(view['shape'], [2, 3, 12, 16])
        self.assertEqual(view['dimensions'], [16, 12, 3])
        direction = np.array(self.project.image.GetDirection()).reshape(4, 4)
        np.testing.assert_allclose(view['direction'], direction[:3, :3].T.reshape(-1))
        values = np.frombuffer(self.client.get(self.prefix + '/volume?time=1').content, dtype='<f4')
        np.testing.assert_array_equal(values, self.project.voxels[1].reshape(-1))
        self.assertEqual(len(self.client.get(self.prefix + '/masks?time=0').content), 3*3*12*16)

    def test_export_4d_is_scalar_and_preserves_all_geometry(self):
        self.brush(t=1, z=2)
        response = self.client.get(self.prefix + '/export?kind=1')
        path = self.root / 'export.nii.gz'; path.write_bytes(response.content)
        image = sitk.ReadImage(str(path))
        self.assertEqual(image.GetDimension(), 4)
        self.assertEqual(image.GetNumberOfComponentsPerPixel(), 1)
        require_same_geometry(self.project.image, image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), self.project.masks[1])

    def test_three_dimensional_export_keeps_reflected_basis(self):
        image = sitk.GetImageFromArray(np.ones((3, 12, 16), dtype=np.float32))
        image.SetSpacing((.8, 1.3, 8.)); image.SetOrigin((2., 3., -4.))
        image.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., 1.))
        path = self.root / 'reflect.nii.gz'; sitk.WriteImage(image, str(path))
        study = self.app.state.store.create(path, 'reflect')
        output = self.root / 'reflect-export.nii.gz'; output.write_bytes(study.export('labelmap')[0])
        actual = sitk.ReadImage(str(output))
        self.assertEqual(actual.GetDimension(), 3)
        require_same_geometry(study.image, actual)

    def test_first_click_undo_redo_restores_prompts_and_prior(self):
        self.assertEqual(self.infer()['state'], 'completed')
        key = (0, 1, 1)
        before = self.project.masks[1].copy()
        prior = self.project.contexts[key]['prior'].copy()
        self.assertEqual(len(self.project.contexts[key]['points']), 1)
        self.post('undo')
        self.assertFalse(self.project.masks[1].any())
        self.assertEqual(self.project.contexts[key]['points'], [])
        self.assertIsNone(self.project.contexts[key]['prior'])
        self.post('redo')
        np.testing.assert_array_equal(self.project.masks[1], before)
        np.testing.assert_array_equal(self.project.contexts[key]['prior'], prior)
        self.assertEqual(len(self.project.contexts[key]['points']), 1)

    def test_brush_uses_edited_mask_as_next_prior_and_undo_restores_click(self):
        self.infer()
        self.brush(points=[[11, 8]])
        context = self.project.contexts[(0, 1, 1)]
        self.assertEqual(context['points'], [])
        np.testing.assert_array_equal(context['prior'], padded_prior(self.project.masks[1][0, 1]))
        self.assertEqual(context['iteration'], 1)
        self.post('undo')
        self.assertEqual(len(self.project.contexts[(0, 1, 1)]['points']), 1)
        self.assertEqual(self.project.masks[1][0, 1, 8, 11], 0)

    def test_brush_overwrites_other_structures_and_undo_restores_both(self):
        self.brush(label=1); before = self.project.masks[1].copy()
        self.brush(label=2)
        self.assertFalse(self.project.masks[1].any())
        self.assertEqual(self.project.overlaps(), 0)
        after = self.project.masks[2].copy()
        self.post('undo')
        np.testing.assert_array_equal(self.project.masks[1], before)
        self.assertFalse(self.project.masks[2].any())
        self.post('redo')
        np.testing.assert_array_equal(self.project.masks[2], after)
        self.assertFalse(self.project.masks[1].any())

    def test_brush_priority_follows_every_selected_organ(self):
        for selected in (1, 2, 3):
            with self.subTest(selected=selected):
                for label in (1, 2, 3):
                    self.project.masks[label].fill(0)
                    if label != selected:
                        self.project.masks[label][0, 1, 5, 6] = 1
                self.brush(label=selected)
                for label in (1, 2, 3):
                    self.assertEqual(self.project.masks[label][0, 1, 5, 6], int(label == selected))
                self.post('undo')
                for label in (1, 2, 3):
                    self.assertEqual(self.project.masks[label][0, 1, 5, 6], int(label != selected))

    def test_single_pixel_brush_overwrites_exactly_one_voxel_for_every_organ(self):
        for selected in (1, 2, 3):
            with self.subTest(selected=selected):
                for label, mask in self.project.masks.items():
                    mask.fill(0)
                    if label != selected:
                        mask[:, :, 4:7, 5:8] = 1
                before = {label: mask.copy() for label, mask in self.project.masks.items()}
                self.brush(label=selected, radius=0)
                after = {}
                for label, mask in self.project.masks.items():
                    expected = before[label].copy()
                    expected[0, 1, 5, 6] = int(label == selected)
                    np.testing.assert_array_equal(mask, expected)
                    self.assertEqual(np.count_nonzero(mask != before[label]), 1)
                    after[label] = expected
                self.post('undo')
                for label in before:
                    np.testing.assert_array_equal(self.project.masks[label], before[label])
                self.post('redo')
                for label in after:
                    np.testing.assert_array_equal(self.project.masks[label], after[label])

    def test_single_pixel_eraser_removes_only_selected_voxel_and_undo_restores_it(self):
        self.project.masks[2][:, :, 4:7, 5:8] = 1
        self.project.masks[1][:, :, 4:7, 9:12] = 1
        self.project.masks[3][:, :, 7:10, 5:8] = 1
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        self.brush(label=2, radius=0, erase=True)
        expected = before[2].copy()
        expected[0, 1, 5, 6] = 0
        np.testing.assert_array_equal(self.project.masks[2], expected)
        self.assertEqual(np.count_nonzero(expected != before[2]), 1)
        for label in (1, 3):
            np.testing.assert_array_equal(self.project.masks[label], before[label])
        self.post('undo')
        np.testing.assert_array_equal(self.project.masks[2], before[2])
        self.post('redo')
        np.testing.assert_array_equal(self.project.masks[2], expected)

    def test_single_pixel_strokes_are_continuous_and_one_pixel_wide(self):
        for start, end, expected in (
            ([3, 5], [9, 5], [(x, 5) for x in range(3, 10)]),
            ([6, 2], [6, 8], [(6, y) for y in range(2, 9)]),
            ([3, 2], [9, 8], [(i + 3, i + 2) for i in range(7)]),
        ):
            with self.subTest(start=start, end=end):
                self.project.masks[1].fill(0)
                self.brush(radius=0, points=[start, end])
                actual = {(int(x), int(y)) for y, x in np.argwhere(self.project.masks[1][0, 1])}
                self.assertEqual(actual, set(expected))
                self.assertEqual(int(self.project.masks[1].sum()), len(expected))

    def test_single_pixel_rounding_edges_and_radius_validation(self):
        for point, expected in (([6.49, 5.49], (6, 5)), ([6.5, 5.5], (7, 6)),
                                ([15.75, 11.75], (15, 11)), ([0, 0], (0, 0))):
            with self.subTest(point=point):
                self.project.masks[1].fill(0)
                self.brush(radius=0, points=[point])
                self.assertEqual(int(self.project.masks[1].sum()), 1)
                self.assertEqual(self.project.masks[1][0, 1, expected[1], expected[0]], 1)
        response = self.client.post(self.prefix + '/brush', json={
            'revision': self.project.revision, 'time': 0, 'slice': 1, 'organ': 1,
            'points': [[6, 5]], 'radius': -1})
        self.assertEqual(response.status_code, 422)

    def test_single_pixel_brush_preserves_locked_voxel_and_its_neighbors(self):
        self.project.masks[1][0, 1, 4:7, 5:8] = 1
        self.post('lock', organ=1, locked=True)
        before = self.project.masks[1].copy()
        self.brush(label=2, radius=0)
        np.testing.assert_array_equal(self.project.masks[1], before)
        self.assertFalse(self.project.masks[2].any())

    def test_legacy_overlap_export_and_slice_scoped_resolution(self):
        self.infer(label=1, mode='organ')
        self.infer(label=3, mode='organ')
        self.project.masks[1][1, 2, 5, 6] = 1
        self.project.masks[3][1, 2, 5, 6] = 1
        self.assertGreater(self.project.overlaps(), 0)
        self.assertEqual(self.client.get(self.prefix + '/export?kind=labelmap').status_code, 400)
        response = self.client.get(self.prefix + '/export?kind=bundle')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {'RV.nii.gz', 'MYO.nii.gz', 'LV.nii.gz', 'annotation.json'})
        self.post('resolve-overlaps', time=0, slice=1, priority=3)
        self.assertEqual(self.project.overlaps(0), 0)
        self.assertEqual(self.project.overlaps(1), 1)
        self.assertFalse(self.project.masks[1][0].any())
        self.post('undo'); self.assertGreater(self.project.overlaps(), 0)

    def test_brush_overwrite_clears_losing_prompts_and_only_touches_current_layer(self):
        self.infer(label=1)
        prior = self.project.contexts[(0, 1, 1)]['prior'].copy()
        self.project.masks[1][0, 0, 5, 6] = 1
        self.project.masks[3][1, 1, 5, 6] = 1
        self.brush(label=2, radius=3)
        self.assertEqual(self.project.contexts[(0, 1, 1)]['points'], [])
        self.assertFalse(self.project.masks[1][0, 1].any())
        self.assertEqual(self.project.masks[1][0, 0, 5, 6], 1)
        self.assertEqual(self.project.masks[3][1, 1, 5, 6], 1)
        np.testing.assert_array_equal(self.project.contexts[(0, 1, 1)]['prior'], padded_prior(self.project.masks[1][0, 1]))
        self.post('undo')
        self.assertEqual(len(self.project.contexts[(0, 1, 1)]['points']), 1)
        np.testing.assert_array_equal(self.project.contexts[(0, 1, 1)]['prior'], prior)

    def probability_predictor(self):
        probabilities = {}
        for label in (1, 2, 3):
            probabilities[label] = np.full((12, 16), .1, dtype=np.float32)
        for label, values in {1: (.7, .8, .9, .5), 2: (.95, .7, .9, .5), 3: (.8, .9, .2, .5)}.items():
            for (y, x), value in zip(((5, 6), (2, 2), (1, 1), (8, 11)), values):
                probabilities[label][y, x] = value
        def predict(image, points, label, mode, prior, iteration):
            self.assertEqual(points, [])
            self.assertEqual(mode, 'organ')
            self.assertIsNone(prior)
            probability = probabilities[label]
            mask = (probability > .5).astype(np.uint8)
            return {'mask': mask, 'probability': probability, 'prior': padded_prior(probability), 'iteration': 1}
        self.app.state.model.predictor = predict

    def test_segment_all_uses_native_probabilities_threshold_ties_and_one_layer(self):
        self.probability_predictor()
        self.brush(label=1, t=0, z=0, points=[[6, 5]], radius=0)
        self.brush(label=3, t=1, z=1, points=[[6, 5]], radius=0)
        revision = self.project.revision
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        result = self.wait_job(self.post('segment-all', time=0, slice=1)['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(self.project.revision, revision + 1)
        self.assertEqual(self.project.overlaps(0), 0)
        self.assertEqual(self.project.masks[2][0, 1, 5, 6], 1)
        self.assertEqual(self.project.masks[3][0, 1, 2, 2], 1)
        self.assertEqual(self.project.masks[1][0, 1, 1, 1], 1)
        for label, mask in self.project.masks.items():
            self.assertEqual(mask[0, 1, 8, 11], 0)
            np.testing.assert_array_equal(mask[0, 0], before[label][0, 0])
            np.testing.assert_array_equal(mask[0, 2], before[label][0, 2])
            np.testing.assert_array_equal(mask[1], before[label][1])
            self.assertEqual(self.project.contexts[(0, 1, label)]['points'], [])
        after = {label: mask.copy() for label, mask in self.project.masks.items()}
        self.post('undo')
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])
        self.post('redo')
        for label in after:
            np.testing.assert_array_equal(self.project.masks[label], after[label])
        self.post('save')
        reopened = ProjectStore(self.root / 'work/projects').get(self.id)
        self.assertEqual(reopened.overlaps(0), 0)
        for label in after:
            np.testing.assert_array_equal(reopened.masks[label], after[label])
            np.testing.assert_array_equal(reopened.contexts[(0, 1, label)]['prior'], self.project.contexts[(0, 1, label)]['prior'])

    def test_segment_all_preserves_locked_pixels_and_undo_works_with_other_lock(self):
        self.brush(label=1)
        self.post('lock', organ=1, locked=True)
        protected = self.project.masks[1].copy()
        self.probability_predictor()
        result = self.wait_job(self.post('segment-all', time=0, slice=1)['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['protected_voxels'], 1)
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertEqual(self.project.overlaps(), 0)
        self.assertEqual(self.project.masks[2][0, 1, 5, 6], 0)
        self.post('undo')
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertFalse(self.project.masks[2].any())

    def test_segment_all_failure_is_atomic(self):
        original = self.predictor.__call__
        def fail_second(image, points, label, mode, prior, iteration):
            if label == 2:
                raise RuntimeError('Second organ failed')
            return original(image, points, label, mode, prior, iteration)
        self.app.state.model.predictor = fail_second
        result = self.wait_job(self.post('segment-all', time=0, slice=1)['id'])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))
        self.assertEqual(self.project.undo_stack, [])
        self.assertIsNone(self.project.pending_job)

    def test_all_layers_predicts_independently_and_restores_history_with_one_undo(self):
        self.infer(label=3)
        self.infer(label=1, t=1, z=2)
        original_masks = {label: mask.copy() for label, mask in self.project.masks.items()}
        other_prior = self.project.contexts[(1, 2, 1)]['prior'].copy()
        old_prior = self.project.contexts[(0, 1, 3)]['prior'].copy()
        original_revision = self.project.revision
        self.probability_predictor()
        result = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['completed_slices'], 3)
        self.assertEqual(result['total_slices'], 3)
        self.assertEqual(self.project.revision, original_revision + 1)
        self.assertEqual(self.project.overlaps(0), 0)
        for z in range(3):
            self.assertEqual(self.project.masks[2][0, z, 5, 6], 1)
            self.assertEqual(self.project.masks[3][0, z, 2, 2], 1)
            self.assertEqual(self.project.masks[1][0, z, 1, 1], 1)
            for label in (1, 2, 3):
                self.assertEqual(self.project.contexts[(0, z, label)]['points'], [])
        for label in original_masks:
            np.testing.assert_array_equal(self.project.masks[label][1], original_masks[label][1])
        np.testing.assert_array_equal(self.project.contexts[(1, 2, 1)]['prior'], other_prior)
        self.post('undo')
        for label in original_masks:
            np.testing.assert_array_equal(self.project.masks[label], original_masks[label])
        self.assertEqual(len(self.project.contexts[(0, 1, 3)]['points']), 1)
        np.testing.assert_array_equal(self.project.contexts[(0, 1, 3)]['prior'], old_prior)
        self.assertNotIn((0, 0, 2), self.project.contexts)
        self.post('redo')
        self.assertEqual(self.project.overlaps(0), 0)
        self.post('save')
        reopened = ProjectStore(self.root / 'work/projects').get(self.id)
        for label in original_masks:
            np.testing.assert_array_equal(reopened.masks[label], self.project.masks[label])
            np.testing.assert_array_equal(reopened.masks[label][1], original_masks[label][1])
        self.assertEqual(reopened.contexts[(1, 2, 1)]['points'], self.project.contexts[(1, 2, 1)]['points'])

    def test_all_layers_uses_each_source_plane_without_other_layer_clicks_or_prior(self):
        self.infer(label=1)
        self.predictor.calls.clear()
        result = self.wait_job(self.post('segment-all-layers', time=1)['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(len(self.predictor.calls), 9)
        for index, call in enumerate(self.predictor.calls):
            np.testing.assert_array_equal(call['image'], self.project.voxels[1, index // 3])
            self.assertEqual(call['label'], index % 3 + 1)
            self.assertEqual(call['points'], [])
            self.assertIsNone(call['prior'])
            self.assertEqual(call['iteration'], 0)
            self.assertEqual(call['mode'], 'organ')
        self.assertEqual(len(self.project.contexts[(0, 1, 1)]['points']), 1)

    def test_all_layers_protects_locked_structure_and_undo_with_lock_is_allowed(self):
        for z in range(3):
            self.brush(label=1, z=z)
        self.post('lock', organ=1, locked=True)
        protected = self.project.masks[1].copy()
        self.probability_predictor()
        result = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['protected_voxels'], 3)
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertEqual(self.project.overlaps(0), 0)
        self.post('undo')
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertFalse(self.project.masks[2].any())
        self.assertFalse(self.project.masks[3].any())

    def test_all_layers_late_failure_does_not_publish_completed_layers(self):
        self.brush(label=3, z=0)
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        revision = self.project.revision
        original = self.predictor.__call__
        def fail_late(image, points, label, mode, prior, iteration):
            if len(self.predictor.calls) == 6:
                raise RuntimeError('Third layer failed')
            return original(image, points, label, mode, prior, iteration)
        self.app.state.model.predictor = fail_late
        result = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['completed_slices'], 2)
        self.assertEqual(self.project.revision, revision)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])
        self.assertEqual(len(self.project.undo_stack), 1)

    def test_all_layers_cancel_after_completed_layer_preserves_all_masks(self):
        entered = threading.Event()
        original = self.predictor.__call__
        def block_second_layer(image, points, label, mode, prior, iteration):
            if len(self.predictor.calls) == 3:
                entered.set()
                if not self.predictor.release.wait(5):
                    raise RuntimeError('Test barrier timed out')
            return original(image, points, label, mode, prior, iteration)
        self.app.state.model.predictor = block_second_layer
        job = self.post('segment-all-layers', time=0)
        self.assertTrue(entered.wait(5))
        self.assertEqual(self.client.get('/api/jobs/' + job['id']).json()['completed_slices'], 1)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        self.predictor.release.set()
        deadline = time.monotonic() + 5
        while self.app.state.jobs.active_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))
        self.assertEqual(self.project.undo_stack, [])

    def test_all_layers_rejects_invalid_scope_locks_and_overlapping_locked_masks(self):
        response = self.client.post(self.prefix + '/segment-all-layers', json={'revision': 0, 'time': 99})
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.project.pending_job)
        for label in (1, 2, 3):
            self.post('lock', organ=label, locked=True)
        response = self.client.post(self.prefix + '/segment-all-layers', json={'revision': self.project.revision, 'time': 0})
        self.assertEqual(response.status_code, 400)
        self.post('lock', organ=3, locked=False)
        self.project.masks[1][0, 2, 5, 6] = 1
        self.project.masks[2][0, 2, 5, 6] = 1
        response = self.client.post(self.prefix + '/segment-all-layers', json={'revision': self.project.revision, 'time': 0})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.project.overlaps(0), 1)

    def test_cancel_all_blocks_device_switch_until_compute_finishes(self):
        self.predictor.block = True
        job = self.post('segment-all', time=0, slice=1)
        self.assertTrue(self.predictor.entered.wait(5))
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        response = self.client.post('/api/model/device', json={'device': 'cpu'})
        self.assertEqual(response.status_code, 400)
        self.predictor.release.set()
        deadline = time.monotonic() + 5
        while self.app.state.jobs.active_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))
        self.assertEqual(len(self.predictor.calls), 1)
        self.assertEqual(self.client.post('/api/model/device', json={'device': 'cpu'}).status_code, 200)

    def test_device_selection_reports_host_and_preserves_annotations(self):
        self.infer()
        mask = self.project.masks[1].copy()
        revision = self.project.revision
        response = self.client.post('/api/model/device', json={'device': 'cpu'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['requested_device'], 'cpu')
        self.assertTrue(response.json()['hostname'])
        np.testing.assert_array_equal(self.project.masks[1], mask)
        self.assertEqual(self.project.revision, revision)
        self.assertEqual(len(self.project.contexts[(0, 1, 1)]['points']), 1)
        self.assertEqual(self.client.post('/api/model/device', json={'device': 'cuda:999'}).status_code, 400)
        self.assertEqual(self.client.post('/api/model/device', json={'device': '../cpu'}).status_code, 400)

    def test_locks_protect_other_structures_and_reject_target_edit(self):
        self.brush(label=1)
        self.post('lock', organ=1, locked=True)
        self.brush(label=3)
        self.assertFalse(self.project.masks[3].any())
        self.assertEqual(self.infer(label=3)['state'], 'completed')
        self.assertEqual(self.project.overlaps(), 0)
        response = self.client.post(self.prefix + '/brush', json={'revision':self.project.revision,
                'time':0, 'slice':1, 'organ':1, 'points':[[6,5]], 'radius':1})
        self.assertEqual(response.status_code, 400)

    def test_time_slice_structure_histories_are_isolated(self):
        self.infer(t=0, z=1, label=1)
        self.infer(t=1, z=1, label=1)
        self.infer(t=0, z=2, label=2)
        self.assertIsNone(self.predictor.calls[1]['prior'])
        self.assertIsNone(self.predictor.calls[2]['prior'])
        self.assertEqual(len(self.project.contexts), 3)
        self.assertFalse(self.project.masks[3].any())

    def test_save_restore_includes_probabilities_points_masks_locks_and_reference(self):
        self.infer()
        self.post('lock', organ=2, locked=True)
        self.post('save')
        other = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(other.revision, self.project.revision)
        self.assertFalse(other.view()['dirty'])
        np.testing.assert_array_equal(other.masks[1], self.project.masks[1])
        np.testing.assert_array_equal(other.contexts[(0,1,1)]['prior'], self.project.contexts[(0,1,1)]['prior'])
        self.assertEqual(other.contexts[(0,1,1)]['points'], self.project.contexts[(0,1,1)]['points'])
        self.assertTrue(other.locks[2])

    def test_geometry_mismatch_rejected_without_any_mutation(self):
        self.brush()
        image = sitk.GetImageFromArray(self.project.masks[1], isVector=False)
        image.CopyInformation(self.project.image)
        spacing = list(image.GetSpacing()); spacing[2] += .2; image.SetSpacing(spacing)
        revision = self.project.revision
        self.assertEqual(self.annotation(image).status_code, 400)
        self.assertEqual(self.project.revision, revision)

    def test_reference_import_never_populates_editable_masks(self):
        values = np.ones(self.project.voxels.shape, dtype=np.uint8)
        image = sitk.GetImageFromArray(values, isVector=False); image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image, target='reference').status_code, 200)
        self.assertTrue(self.project.reference.any())
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_binary_import_only_changes_selected_structure(self):
        self.brush(label=1); before = self.project.masks[1].copy()
        values = np.zeros(self.project.voxels.shape, dtype=np.uint8); values[1,2,3:6,8:11] = 1
        image = sitk.GetImageFromArray(values, isVector=False); image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image, binary=3).status_code, 200)
        np.testing.assert_array_equal(self.project.masks[1], before)
        np.testing.assert_array_equal(self.project.masks[3], values)

    def test_cancellation_does_not_commit_mask_or_append_point(self):
        self.predictor.block = True
        job = self.infer(wait=False); self.assertTrue(self.predictor.entered.wait(3))
        response = self.client.post('/api/jobs/' + job['id'] + '/cancel')
        self.assertEqual(response.status_code, 200)
        self.predictor.release.set(); self.app.state.jobs.close()
        self.assertEqual(self.project.revision, 0)
        self.assertEqual(self.project.contexts[(0,1,1)]['points'], [])
        self.assertFalse(self.project.masks[1].any())
        self.assertIsNone(self.project.pending_job)

    def test_changed_revision_discards_stale_inference(self):
        self.predictor.block = True
        job = self.infer(wait=False); self.assertTrue(self.predictor.entered.wait(3))
        self.post('lock', organ=2, locked=True)
        self.predictor.release.set()
        self.assertEqual(self.wait_job(job['id'])['state'], 'stale')
        self.assertFalse(self.project.masks[1].any())
        self.assertEqual(self.project.revision, 1)

    def test_failed_inference_leaves_state_unchanged(self):
        self.predictor.fail = True
        with self.assertLogs('server.inference', level='ERROR'):
            result = self.infer()
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(self.project.masks[1].any())
        self.assertEqual(self.project.revision, 0)
        self.assertEqual(self.project.contexts[(0,1,1)]['points'], [])

    def test_mode_switch_organ_to_pure_click_discards_baseline(self):
        self.infer(mode='organ')
        self.infer(mode='click')
        self.assertIsNone(self.predictor.calls[-1]['prior'])
        self.assertEqual(self.predictor.calls[-1]['iteration'], 0)
        self.infer(mode='organ_click')
        self.assertIsNone(self.predictor.calls[-1]['prior'])
        self.assertEqual(len(self.predictor.calls[-1]['points']), 1)

    def test_stale_revision_and_invalid_click_cannot_mutate_state(self):
        self.brush()
        response = self.client.post(self.prefix + '/clear-slice', json={'revision':0,'time':0,'slice':1,'organ':1})
        self.assertEqual(response.status_code, 400)
        response = self.client.post(self.prefix + '/inference', json={'revision':1,'time':0,'slice':1,'organ':1,
                 'mode':'click','point':{'x':99,'y':5,'positive':True}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.project.revision, 1)
        self.assertTrue(self.project.masks[1].any())

    def test_distinct_file_identity_even_for_same_name_and_dimensions(self):
        other = self.app.state.store.create(self.path, 'oblique4d.nii.gz')
        self.assertNotEqual(other.id, self.id)
        self.brush()
        self.assertFalse(other.masks[1].any())
        self.assertEqual(self.client.get('/api/projects/not-a-uuid').status_code, 404)

    def test_evaluated_demo_predictions_are_distinct_from_gt_and_undoable(self):
        name = 'example_4d.nii.gz'
        reference_dir = self.library_root / 'demo-data/references'
        predictions_dir = self.library_root / 'demo-data/predictions'
        reference_dir.mkdir(); predictions_dir.mkdir()
        source = sitk.ReadImage(str(self.path))
        reference = np.zeros(self.project.voxels.shape, dtype=np.uint8)
        reference[:, :, 1:3, 1:3] = 1
        image = sitk.GetImageFromArray(reference, isVector=False); image.CopyInformation(source)
        sitk.WriteImage(image, str(reference_dir / 'example_4d_gt.nii.gz'))
        for label, short in ((1, 'RV'), (2, 'MYO'), (3, 'LV')):
            mask = np.zeros_like(reference); mask[:, :, 5:7, label * 3:label * 3 + 2] = 1
            image = sitk.GetImageFromArray(mask, isVector=False); image.CopyInformation(source)
            sitk.WriteImage(image, str(predictions_dir / f'example_4d_{short}_single_click.nii.gz'))
        (self.library_root / 'demo-data/provenance.json').write_text(json.dumps([{'name': name}]))
        (predictions_dir / 'evaluation.json').write_text(json.dumps({
            'protocol': 'synthetic_fixture', 'source_report': 'generated by test',
            'cases': [{'file': name, 'organs': {short: {'single_click_3d_dice': 0.}
                                             for short in ('RV', 'MYO', 'LV')}}]}))
        response = self.client.post('/api/demos/open', json={'name': name})
        self.assertEqual(response.status_code, 200, response.text)
        demo = self.app.state.store.get(response.json()['id'])
        self.assertFalse(any(mask.any() for mask in demo.masks.values()))
        self.assertTrue(demo.reference.any())
        response = self.client.post('/api/projects/' + demo.id + '/demo-predictions', json={'revision':0})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(mask.any() for mask in demo.masks.values()))
        self.assertFalse(np.array_equal(demo.masks[1], demo.reference == 1))
        restored = {label: mask.copy() for label, mask in demo.masks.items()}
        demo.undo(); self.assertFalse(any(mask.any() for mask in demo.masks.values()))
        demo.undo(redo=True)
        for label in restored:
            np.testing.assert_array_equal(demo.masks[label], restored[label])

    def test_invalid_navigation_cannot_partially_apply_undo(self):
        self.brush()
        response = self.client.post(self.prefix + '/undo?time=99', json={'revision':1})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.project.revision, 1)
        self.assertTrue(self.project.masks[1].any())
        self.assertEqual(len(self.project.undo_stack), 1)

    def test_cancel_then_new_click_commits_only_the_new_request(self):
        self.predictor.block = True
        first = self.infer(wait=False); self.assertTrue(self.predictor.entered.wait(3))
        self.client.post('/api/jobs/' + first['id'] + '/cancel').raise_for_status()
        second = self.infer(point={'x':11.,'y':8.,'positive':True}, wait=False)
        self.predictor.release.set()
        self.assertEqual(self.wait_job(second['id'])['state'], 'completed')
        self.assertEqual(self.wait_job(first['id'])['state'], 'cancelled')
        self.assertEqual(self.project.masks[1][0,1,5,6], 0)
        self.assertEqual(self.project.masks[1][0,1,8,11], 1)
        self.assertEqual(len(self.project.contexts[(0,1,1)]['points']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
