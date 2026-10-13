"""Temporal references, sparse transactions, and safe refinement; no CUDA needed."""
import threading
import time
import unittest

import numpy as np
import SimpleITK as sitk

import test_platform as fixtures
from server.annotation_progress import summary
from server.domain import ProjectStore
from server.inference import padded_prior


class TemporalPropagationTests(unittest.TestCase):
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job

    def setUp(self):
        fixtures.PlatformTests.setUp(self)
        values = np.arange(6*3*12*16, dtype=np.float32).reshape(6, 3, 12, 16)
        image = sitk.GetImageFromArray(values, isVector=False)
        image.SetSpacing(self.project.image.GetSpacing())
        image.SetOrigin(self.project.image.GetOrigin())
        image.SetDirection(self.project.image.GetDirection())
        path = self.root / 'six-frames.nii.gz'
        sitk.WriteImage(image, str(path))
        self.project = self.app.state.store.create(path, 'six-frames')
        self.id = self.project.id
        self.prefix = '/api/projects/' + self.id
        self.references = [0, 2, 4]
        for t in self.references:
            for label in (1, 2, 3):
                self.brush(label=label, t=t, points=[[label * 3, 2]], radius=0)
        self.reference_calls = []
        self.predictor.predict_reference = self.reference_predictor

    def reference_predictor(self, image, references, label):
        self.reference_calls.append({'image': image.copy(), 'references': [(p.copy(), m.copy()) for p, m in references], 'label': label})
        self.predictor.entered.set()
        if self.predictor.block and not self.predictor.release.wait(10):
            raise RuntimeError('Reference test barrier timed out')
        if self.predictor.fail:
            raise RuntimeError('Deliberate reference test failure')
        probability = np.full(image.shape, .05, np.float32)
        probability[4:7, 4:9] = .5 + .1 * label
        probability[label, label] = .99
        mask = probability > .5
        return {'mask': mask, 'probability': probability, 'prior': padded_prior(mask), 'iteration': 1}

    def request(self, **details):
        return {'slice': 1, 'reference_times': self.references, 'organs': [1, 2, 3], **details}

    def propagate(self, **details):
        job = self.post('propagate-time', **self.request(**details))
        return self.wait_job(job['id'])

    def drain(self):
        deadline = time.monotonic() + 10
        while self.app.state.jobs.active_tasks and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.app.state.jobs.active_tasks, 0)

    def copy_masks(self):
        return {label: mask.copy() for label, mask in self.project.masks.items()}

    def assert_masks(self, expected):
        for label, mask in expected.items():
            np.testing.assert_array_equal(self.project.masks[label], mask)

    def test_three_references_are_frozen_other_layers_and_reference_history_are_preserved(self):
        before = self.copy_masks()
        progress = summary(self.project)
        self.post('save')
        reads = self.project.prior_store.reads
        preview = self.post('propagation-preview', **self.request())
        self.assertEqual(preview['total_frames'], 3)
        self.assertEqual(preview['total_predictions'], 9)
        result = self.propagate()
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['changed_times'], [1, 3, 5])
        self.assertEqual(len(self.reference_calls), 9)
        for index, call in enumerate(self.reference_calls):
            t = [1, 3, 5][index // 3]
            np.testing.assert_array_equal(call['image'], self.project.voxels[t, 1])
            self.assertEqual(len(call['references']), 3)
            for (plane, mask), ref in zip(call['references'], self.references):
                np.testing.assert_array_equal(plane, self.project.voxels[ref, 1])
                np.testing.assert_array_equal(mask, before[call['label']][ref, 1])
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][self.references], before[label][self.references])
            np.testing.assert_array_equal(self.project.masks[label][:, [0, 2]], before[label][:, [0, 2]])
        self.assertEqual(summary(self.project)['edited_labels'], progress['edited_labels'])
        self.assertEqual(summary(self.project)['edited_at'], progress['edited_at'])
        self.assertEqual(self.project.prior_store.reads, reads)
        self.assertEqual(set(self.project.dirty_layers), {(t, 1, label) for t in (1, 3, 5) for label in (1, 2, 3)})
        self.assertEqual(self.project.overlaps(), 0)
        self.assertEqual(self.project.masks[3][1, 1, 5, 6], 1)

    def test_manual_targets_locked_and_unselected_pixels_are_protected(self):
        self.brush(label=2, t=1, points=[[6, 5]], radius=0)
        self.brush(label=3, t=3, points=[[7, 5]], radius=0)
        self.post('lock', organ=3, locked=True)
        before = self.copy_masks()
        preview = self.post('propagation-preview', **self.request())
        self.assertIn({'time': 1, 'organ': 2, 'reason': 'edited'}, preview['skipped'])
        self.assertIn({'time': 3, 'organ': 3, 'reason': 'locked'}, preview['skipped'])
        self.assertEqual(self.propagate()['state'], 'completed')
        np.testing.assert_array_equal(self.project.masks[3], before[3])
        np.testing.assert_array_equal(self.project.masks[2][1, 1], before[2][1, 1])
        self.assertEqual(self.project.masks[1][1, 1, 5, 6], 0)
        self.assertEqual(self.project.masks[1][3, 1, 5, 7], 0)
        self.assertEqual(self.project.masks[2][3, 1, 5, 7], 0)
        self.assertEqual(self.project.overlaps(), 0)
        self.post('undo')
        self.assert_masks(before)

    def test_explicit_overwrite_preserves_reference_timestamps_and_undo_redo_exactly(self):
        self.brush(label=2, t=1, points=[[6, 5]], radius=0)
        before = self.copy_masks()
        progress = summary(self.project)
        result = self.propagate(overwrite_edited=True)
        self.assertEqual(result['state'], 'completed', result)
        after = self.copy_masks()
        after_progress = summary(self.project)
        self.assertEqual(after_progress['edited_labels'][4], 0)
        for t in self.references:
            self.assertEqual(after_progress['edited_labels'][t*3+1], 7)
        self.post('undo')
        self.assert_masks(before)
        for key in ('edited_labels', 'edited_at', 'mask_labels'):
            self.assertEqual(summary(self.project)[key], progress[key])
        self.post('redo')
        self.assert_masks(after)
        for key in ('edited_labels', 'edited_at', 'mask_labels'):
            self.assertEqual(summary(self.project)[key], after_progress[key])

    def test_next_negative_click_in_both_modes_receives_propagated_prior(self):
        self.assertEqual(self.propagate()['state'], 'completed')
        for mode, t in [('click', 1), ('organ_click', 3)]:
            prior = self.project.contexts[(t, 1, 2)]['prior'].copy()
            result = self.infer(label=2, mode=mode, t=t, point={'x': 2, 'y': 2, 'positive': False})
            self.assertEqual(result['state'], 'completed', result)
            call = self.predictor.calls[-1]
            np.testing.assert_array_equal(call['prior'], prior)
            self.assertEqual(call['iteration'], 1)
            self.assertEqual(call['points'], [{'x': 2., 'y': 2., 'positive': False}])

    def test_missing_masks_skip_organ_and_keep_existing_mask_as_protected(self):
        for t in self.references:
            self.post('clear-slice', time=t, slice=1, organ=3)
        self.brush(label=3, t=1, points=[[6, 5]], radius=0)
        before = self.project.masks[3].copy()
        preview = self.post('propagation-preview', **self.request(overwrite_edited=True))
        self.assertEqual(preview['total_predictions'], 6)
        self.assertIn({'time': 1, 'organ': 3, 'reason': 'no_reference_mask'}, preview['skipped'])
        result = self.propagate(overwrite_edited=True)
        self.assertEqual(result['state'], 'completed', result)
        np.testing.assert_array_equal(self.project.masks[3], before)
        self.assertFalse(self.project.masks[1][1, 1, 5, 6])
        self.assertFalse(self.project.masks[2][1, 1, 5, 6])

    def test_one_and_two_references_use_all_other_times_and_only_available_supports(self):
        for refs in [[0], [0, 2]]:
            self.reference_calls.clear()
            result = self.propagate(reference_times=refs, overwrite_edited=True, organs=[1])
            self.assertEqual(result['state'], 'completed', result)
            self.assertEqual(result['total_frames'], 6-len(refs))
            self.assertTrue(all(len(call['references']) == len(refs) for call in self.reference_calls))
            self.post('undo')

    def test_explicit_target_selection_never_overwrites_reference(self):
        before = self.copy_masks()
        result = self.propagate(target_times=[0, 3], organs=[2])
        self.assertEqual(result['changed_times'], [3])
        self.assertEqual(len(self.reference_calls), 1)
        np.testing.assert_array_equal(self.project.masks[2][[0, 1, 2, 4, 5]], before[2][[0, 1, 2, 4, 5]])

    def test_cancel_stale_and_failure_do_not_publish_any_target(self):
        for outcome in ('cancelled', 'stale', 'failed'):
            before = self.copy_masks()
            revision = self.project.revision
            self.predictor.block = outcome != 'failed'
            self.predictor.fail = outcome == 'failed'
            self.predictor.entered.clear()
            self.predictor.release.clear()
            job = self.post('propagate-time', **self.request())
            self.assertTrue(self.predictor.entered.wait(5))
            if outcome == 'cancelled':
                self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status()
                self.assertEqual(self.client.post('/api/model/device', json={'device': 'cpu'}).status_code, 400)
            elif outcome == 'stale':
                with self.project.lock:
                    self.project.revision += 1
            self.predictor.release.set()
            self.assertEqual(self.wait_job(job['id'])['state'], outcome)
            self.drain()
            self.assert_masks(before)
            self.assertEqual(self.project.revision, revision + int(outcome == 'stale'))

    def test_saved_propagation_roundtrips_context_history_and_never_reads_other_priors(self):
        self.infer(label=1, t=5, z=2)
        self.post('save')
        reads = self.project.prior_store.reads
        self.assertEqual(self.propagate()['state'], 'completed')
        after = self.copy_masks()
        self.post('save')
        self.assertEqual(self.project.prior_store.reads, reads)
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(reopened.prior_store.reads, 0)
        for label in after:
            np.testing.assert_array_equal(reopened.masks[label], after[label])
        self.assertEqual(summary(reopened), summary(self.project))
        self.assertEqual(reopened.contexts[(1, 1, 2)]['propagation']['reference_times'], [0, 2, 4])
        self.assertEqual(reopened.contexts[(1, 1, 2)]['iteration'], 1)
        backup = self.client.get(self.prefix + '/export?kind=study')
        restored = self.client.post('/api/projects/import-backup', files={'file': ('study.zip', backup.content)})
        self.assertEqual(restored.status_code, 200, restored.text)
        imported = self.app.state.store.get(restored.json()['id'])
        for label in after:
            np.testing.assert_array_equal(imported.masks[label], after[label])

    def test_request_validation_and_completed_study_are_nonmutating(self):
        before = self.copy_masks()
        invalid = [dict(reference_times=[]), dict(reference_times=[0, 0]), dict(reference_times=[0, 1, 2, 3]),
                   dict(reference_times=[6]), dict(reference_times=[True]), dict(slice=3),
                   dict(organs=[]), dict(organs=[1, 1]), dict(organs=[4]), dict(target_times=[]),
                   dict(target_times=[6]), dict(target_times=[1, 1])]
        for patch in invalid:
            response = self.client.post(self.prefix + '/propagate-time', json={'revision': self.project.revision, **self.request(**patch)})
            self.assertIn(response.status_code, (400, 422), response.text)
        self.client.post(self.prefix + '/notes', json={'revision': 0, 'text': 'Reviewed', 'review_status': 'complete'}).raise_for_status()
        response = self.client.post(self.prefix + '/propagate-time', json={'revision': self.project.revision, **self.request()})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Reopen', response.text)
        self.assert_masks(before)
        self.assertEqual(self.reference_calls, [])

    def test_multiple_workers_use_frozen_references_and_publish_one_transaction(self):
        guard = threading.Lock()
        gate = threading.Barrier(2)
        startup = threading.Barrier(2)
        calls = []
        predict = self.reference_predictor
        class Worker:
            checkpoint_sha256 = 'shared-test-checkpoint'
            def __init__(self, gpu_id):
                self.gpu_id = gpu_id
            def predict_reference(self, task_id, image, labels, references):
                with guard:
                    calls.append((task_id, self.gpu_id))
                    first = len(calls) <= 2
                if first:
                    gate.wait(5)
                return {label: predict(image, references[label], label) for label in labels}
        class Pool:
            def validate(self, ids):
                pass
            def get(self, gpu_id):
                startup.wait(3)  # Propagation must use concurrent initialization too.
                return Worker(gpu_id)
            def close(self):
                pass
        self.app.state.jobs.gpu_pool = Pool()
        revision = self.project.revision
        result = self.propagate(gpu_ids=[2, 3])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(self.project.revision, revision + 1)
        self.assertEqual(result['checkpoint_sha256'], 'shared-test-checkpoint')
        self.assertEqual({gpu for _, gpu in calls}, {2, 3})
        self.assertEqual(result['completed_frames'], 3)
        self.assertEqual(sum(result['gpu_completed'].values()), 3)
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(result['ready_gpu_ids'], [2, 3])
        self.assertEqual(result['preparing_gpu_ids'], [])
        self.assertEqual(result['result_visibility'], 'on_completion')


if __name__ == '__main__':
    unittest.main()
