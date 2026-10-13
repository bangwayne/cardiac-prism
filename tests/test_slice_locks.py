"""Per-(time, layer) protection, legacy migration and sparse batch execution."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_platform as fixtures
import test_refinement_controls as refinement
import test_all_time_frames as all_frames
import test_temporal_propagation as temporal
from server.domain import ProjectStore


class SliceLockTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job
    annotation = fixtures.PlatformTests.annotation
    paired_fixture = refinement.RefinementTests.paired_fixture
    fake_pool = all_frames.AllTimeFramesTests.fake_pool

    def protect(self, times=(0,), slices=(1,), locked=True):
        return self.post('slice-locks', times=list(times), slices=list(slices), locked=locked)

    def masks(self):
        return {label: mask.copy() for label, mask in self.project.masks.items()}

    def assert_masks(self, expected):
        for label in expected:
            np.testing.assert_array_equal(self.project.masks[label], expected[label])

    def chunks(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.project.directory.glob('*.npz')}

    def test_cartesian_selection_and_partial_unlock_preserve_native_metadata_only(self):
        self.infer(label=2)
        self.post('save')
        chunks, reads = self.chunks(), self.project.prior_store.reads
        before = self.masks()
        contexts = dict(self.project.contexts)
        result = self.protect(times=(0, 1), slices=(1, 2))
        self.assertEqual(result['locked_slices'], [[0, 1], [0, 2], [1, 1], [1, 2]])
        self.assertEqual(result['locked_frames'], [])
        self.protect(times=(1,), slices=(0,))
        self.assertEqual(self.project.locked_frames, {1})
        result = self.protect(times=(1,), slices=(1,), locked=False)
        self.assertEqual(result['locked_frames'], [])
        self.assertEqual(result['locked_slices'], [[0, 1], [0, 2], [1, 0], [1, 2]])
        self.assert_masks(before)
        self.assertEqual(self.chunks(), chunks)
        self.assertEqual(self.project.prior_store.reads, reads)
        for key, context in contexts.items():
            self.assertIs(self.project.contexts[key], context)
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.locked_slices, self.project.locked_slices)
        self.assertTrue(self.client.get('/api/health').json()['features']['slice_locks'])

    def test_legacy_full_frame_locks_migrate_then_allow_one_layer_to_unlock(self):
        self.infer(t=0, z=1)
        self.post('save')
        path = self.project.directory / 'project.json'
        manifest = json.loads(path.read_text())
        manifest.pop('locked_slices')
        manifest['locked_frames'] = [0]
        path.write_text(json.dumps(manifest))
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.locked_slices, {(0, 0), (0, 1), (0, 2)})
        self.app.state.store.projects[self.id] = self.project = reopened
        chunks = self.chunks()
        self.protect(slices=(1,), locked=False)
        self.assertEqual(self.project.locked_slices, {(0, 0), (0, 2)})
        self.assertEqual(self.project.locked_frames, set())
        self.assertEqual(self.chunks(), chunks)
        self.assertEqual(self.project.prior_store.reads, 0)
        self.assertEqual(self.infer(t=0, z=1)['state'], 'completed')
        self.post('save')
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.locked_slices, {(0, 0), (0, 2)})

    def test_direct_edits_protect_only_selected_plane_and_all_organs(self):
        self.protect()
        revision = self.project.revision
        for label in (1, 2, 3):
            for action, details in [('brush', dict(points=[[2, 2]], radius=0)), ('clear-slice', {}),
                                    ('clear-points', {}), ('inference', dict(mode='organ'))]:
                response = self.client.post(self.prefix + '/' + action,
                    json={'revision': revision, 'time': 0, 'slice': 1, 'organ': label, **details})
                self.assertEqual(response.status_code, 400, response.text)
        for action, details in [('segment-all', {}), ('clear-all-masks-slice', {}),
                                ('resolve-overlaps', {'priority': 2})]:
            response = self.client.post(self.prefix + '/' + action,
                json={'revision': revision, 'time': 0, 'slice': 1, **details})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.project.revision, revision)
        self.assertEqual(self.project.contexts, {})
        self.brush(label=2, t=0, z=0, radius=0)
        self.brush(label=3, t=1, z=1, radius=0)
        self.assertTrue(self.project.masks[2][0, 0].any())
        self.assertTrue(self.project.masks[3][1, 1].any())
        self.assertTrue(all(not mask[0, 1].any() for mask in self.project.masks.values()))

    def test_all_layers_skips_partial_frame_lock_and_undo_keeps_protected_context(self):
        self.infer(label=2, t=0, z=1)
        self.post('save')
        before = self.masks()
        context = self.project.contexts[(0, 1, 2)]
        self.protect()
        self.predictor.calls.clear()
        result = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['target_slices'], [0, 2])
        self.assertEqual(result['skipped_locked_slices'], [[0, 1]])
        self.assertEqual(result['total_slices'], result['completed_slices'], 2)
        self.assertEqual(len(self.predictor.calls), 6)
        for index, call in enumerate(self.predictor.calls):
            np.testing.assert_array_equal(call['image'], self.project.voxels[0, [0, 2][index // 3]])
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0, 1], before[label][0, 1])
            np.testing.assert_array_equal(self.project.masks[label][1], before[label][1])
        self.assertIs(self.project.contexts[(0, 1, 2)], context)
        self.post('undo')
        self.assert_masks(before)
        self.post('redo')
        self.assertIs(self.project.contexts[(0, 1, 2)], context)

    def test_all_time_cpu_and_multi_gpu_process_only_unlocked_pairs_and_match(self):
        self.infer(label=2, t=0, z=1)
        self.post('save')
        before = self.masks()
        context = self.project.contexts[(0, 1, 2)]
        self.protect(slices=(0, 1))
        self.predictor.calls.clear()
        result = self.wait_job(self.post('segment-all-time-frames')['id'])
        expected = [[0, 2], [1, 0], [1, 1], [1, 2]]
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['target_slices'], expected)
        self.assertEqual(result['skipped_locked_slices'], [[0, 0], [0, 1]])
        self.assertEqual(result['slice'], 2)
        self.assertEqual(result['total_slices'], 4)
        self.assertEqual(result['completed_slices'], 4)
        self.assertEqual(result['completed_frames'], 2)
        cpu = self.masks()
        self.assertIs(self.project.contexts[(0, 1, 2)], context)
        self.post('undo')
        self.assert_masks(before)
        counters = self.fake_pool()
        result = self.wait_job(self.post('segment-all-time-frames', gpu_ids=[4, 5])['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['completed_slices'], 4)
        sent = {(int(call.rsplit(':', 2)[1]), int(call.rsplit(':', 1)[1])) for call in counters['calls']}
        self.assertEqual(sent, set(map(tuple, expected)))
        self.assert_masks(cpu)
        self.assertIs(self.project.contexts[(0, 1, 2)], context)
        self.post('undo')
        self.assert_masks(before)
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_bulk_import_baseline_and_clear_skip_exact_pairs_with_scoped_undo(self):
        self.infer(label=2, t=0, z=1)
        self.post('save')
        before = self.masks()
        context = self.project.contexts[(0, 1, 2)]
        self.protect()
        labels, _, _ = self.paired_fixture()
        image = sitk.GetImageFromArray(labels, isVector=False); image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image).status_code, 200)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0, 1], before[label][0, 1])
            np.testing.assert_array_equal(self.project.masks[label][0, 0], labels[0, 0] == label)
        self.assertIs(self.project.contexts[(0, 1, 2)], context)
        self.post('undo')
        self.assert_masks(before)
        with patch('server.app.ROOT', self.root):
            self.post('paired-annotation')
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0, 1], before[label][0, 1])
        self.assertIs(self.project.contexts[(0, 1, 2)], context)
        self.post('clear-all-masks-layer', time=0, slice=1)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0, 1], before[label][0, 1])
            self.assertFalse(self.project.masks[label][1, 1].any())
        self.assertEqual({(t, z) for t, z, _ in self.project.undo_stack[-1].planes}, {(1, 1)})
        self.post('undo')
        self.post('undo')
        self.assert_masks(before)
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_undo_rejects_locked_pair_but_allows_other_layer_in_same_frame(self):
        self.brush(t=0, z=0)
        self.protect()
        self.post('undo')
        self.assertFalse(self.project.masks[1].any())
        self.post('redo')
        self.protect(slices=(0,))
        before = self.masks()
        count = len(self.project.undo_stack)
        response = self.client.post(self.prefix + '/undo', json={'revision': self.project.revision})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.project.undo_stack), count)
        self.assert_masks(before)

    def test_atomic_slice_lock_failures_preserve_clean_and_dirty_state(self):
        self.infer(t=1)
        self.post('save')
        for locked in (True, False):
            if not locked:
                self.protect()
            for dirty in (False, True):
                if dirty:
                    self.brush(t=1, z=0)
                before = {
                    'revision': self.project.revision, 'saved_revision': self.project.saved_revision,
                    'events': copy.deepcopy(self.project.events), 'locked_slices': set(self.project.locked_slices),
                    'undo': [id(value) for value in self.project.undo_stack], 'chunks': self.chunks(),
                    'manifest': (self.project.directory / 'project.json').read_bytes(),
                    'reads': self.project.prior_store.reads,
                }
                with patch('server.domain.os.replace', side_effect=OSError('Injected slice metadata failure')):
                    with patch.object(self.client._transport, 'raise_server_exceptions', False):
                        result = self.client.post(self.prefix + '/slice-locks',
                            json={'revision': self.project.revision, 'times': [0], 'slices': [1], 'locked': locked})
                self.assertEqual(result.status_code, 500)
                for name in ('revision', 'saved_revision', 'events', 'locked_slices'):
                    self.assertEqual(getattr(self.project, name), before[name])
                self.assertEqual([id(value) for value in self.project.undo_stack], before['undo'])
                self.assertEqual(self.chunks(), before['chunks'])
                self.assertEqual((self.project.directory / 'project.json').read_bytes(), before['manifest'])
                self.assertEqual(self.project.prior_store.reads, before['reads'])
            self.post('save')

    def test_slice_lock_request_validation_is_nonmutating(self):
        for values in (dict(times=[]), dict(slices=[]), dict(times=[0, 0]), dict(slices=[1, 1]),
                       dict(times=[2]), dict(slices=[3]), dict(times=[True]), dict(slices=['1'])):
            result = self.client.post(self.prefix + '/slice-locks',
                json={'revision': self.project.revision, 'times': [0], 'slices': [1], 'locked': True, **values})
            self.assertIn(result.status_code, (400, 422), result.text)
        self.assertEqual(self.project.locked_slices, set())
        self.assertEqual(self.project.revision, 0)

    def test_batch_writeback_rechecks_pairs_and_discards_every_result_atomically(self):
        for action, details in [('segment-all-layers', {'time': 0}), ('segment-all-time-frames', {})]:
            self.predictor.block = True
            self.predictor.entered.clear(); self.predictor.release.clear()
            before = self.masks()
            revision = self.project.revision
            result = self.post(action, **details)
            self.assertTrue(self.predictor.entered.wait(5))
            # API changes are refused while pending. Simulate an out-of-band
            # protection change to prove final publication revalidates targets.
            with self.project.lock:
                self.project.locked_slices.add((0, 0))
            self.predictor.release.set()
            result = self.wait_job(result['id'])
            self.assertEqual(result['state'], 'failed', result)
            self.assert_masks(before)
            self.assertEqual(self.project.revision, revision)
            self.assertEqual(self.project.undo_stack, [])
            with self.project.lock:
                self.project.locked_slices.clear()

    def test_3d_lock_protects_one_layer_and_all_layers_processes_neighbors(self):
        path = self.root / 'three.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((3, 12, 16), np.float32)), str(path))
        self.project = self.app.state.store.create(path, 'three.nii.gz')
        self.id = self.project.id; self.prefix = '/api/projects/' + self.id
        self.protect()
        self.assertEqual(self.project.locked_frames, set())
        result = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['target_slices'], [0, 2])
        self.assertTrue(all(not mask[0, 1].any() for mask in self.project.masks.values()))


class PropagationSliceLockTests(unittest.TestCase):
    setUp = temporal.TemporalPropagationTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    wait_job = fixtures.PlatformTests.wait_job
    reference_predictor = temporal.TemporalPropagationTests.reference_predictor
    request = temporal.TemporalPropagationTests.request
    propagate = temporal.TemporalPropagationTests.propagate

    def test_locked_reference_guides_targets_and_locked_target_on_same_layer_is_skipped(self):
        self.post('slice-locks', times=[0, 1], slices=[1], locked=True)
        preview = self.post('propagation-preview', **self.request(overwrite_edited=True))
        self.assertEqual([entry['time'] for entry in preview['eligible']], [3, 5])
        self.assertEqual(preview['skipped_locked_slices'], [[1, 1]])
        self.assertIn({'time': 1, 'organ': 2, 'reason': 'slice_locked'}, preview['skipped'])
        result = self.propagate(overwrite_edited=True)
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['target_slices'], [[3, 1], [5, 1]])
        self.assertEqual(len(self.reference_calls), 6)
        self.assertEqual(self.project.locked_frames, set())
        self.assertTrue(all(not mask[1, 1].any() for mask in self.project.masks.values()))


if __name__ == '__main__':
    unittest.main()
