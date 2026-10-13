"""Whole-time-frame protection across edits, inference, persistence and history."""
import copy
import hashlib
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_platform as fixtures
import test_refinement_controls as refinement
import test_temporal_propagation as temporal
import test_all_time_frames as all_frames
from server.domain import ProjectStore


class FrameLockTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job
    annotation = fixtures.PlatformTests.annotation
    paired_fixture = refinement.RefinementTests.paired_fixture
    fake_pool = all_frames.AllTimeFramesTests.fake_pool

    def chunks(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.project.directory.glob('*.npz')}

    def masks(self):
        return {label: mask.copy() for label, mask in self.project.masks.items()}

    def assert_masks(self, expected):
        for label in expected:
            np.testing.assert_array_equal(self.project.masks[label], expected[label])

    def drain(self):
        deadline = time.monotonic() + 10
        while self.app.state.jobs.active_tasks and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.app.state.jobs.active_tasks, 0)

    def lock(self, times=(0,), locked=True):
        return self.post('frame-locks', times=list(times), locked=locked)

    def test_clean_lock_is_saved_without_loading_priors_or_rewriting_chunks(self):
        self.infer(t=0)
        self.post('save')
        chunks, reads = self.chunks(), self.project.prior_store.reads
        last_save = copy.deepcopy(self.project.last_save)
        context = self.project.contexts[(0, 1, 1)]
        undo = list(self.project.undo_stack)
        result = self.lock()
        self.assertEqual(result['locked_frames'], [0])
        self.assertEqual(result['saved_revision'], result['revision'])
        self.assertEqual(self.chunks(), chunks)
        self.assertEqual(self.project.last_save, last_save)
        self.assertEqual(self.project.prior_store.reads, reads)
        self.assertIs(self.project.contexts[(0, 1, 1)], context)
        self.assertEqual(self.project.undo_stack, undo)
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.locked_frames, {0})
        self.assertTrue(self.client.get('/api/health').json()['features']['frame_locks'])

    def test_dirty_lock_persists_only_protection_and_does_not_claim_edits_saved(self):
        self.brush(t=0, radius=0)
        expected = self.masks()
        chunks, last_save = self.chunks(), copy.deepcopy(self.project.last_save)
        manifest = json.loads((self.project.directory / 'project.json').read_text())
        saved_revision = self.project.saved_revision
        dirty = dict(self.project.dirty_layers)
        result = self.lock()
        self.assertTrue(result['dirty'])
        self.assertEqual(self.project.saved_revision, saved_revision)
        self.assertEqual(self.project.dirty_layers, dirty)
        self.assertEqual(self.chunks(), chunks)
        self.assertEqual(self.project.last_save, last_save)
        self.assert_masks(expected)
        updated = json.loads((self.project.directory / 'project.json').read_text())
        self.assertEqual(updated, {**manifest, 'locked_frames': [0], 'locked_slices': [[0, z] for z in range(3)]})
        self.assertEqual(ProjectStore(self.project.directory.parent).get(self.id).locked_frames, {0})
        self.post('save')
        reopened = ProjectStore(self.project.directory.parent).get(self.id)
        self.assertEqual(reopened.locked_frames, {0})
        for label in expected:
            np.testing.assert_array_equal(reopened.masks[label], expected[label])
        self.assertEqual(self.project.prior_store.reads, 0)

    def assert_failed_frame_change_is_atomic(self, locked):
        expected = self.masks()
        project = self.project
        before = {
            'revision': project.revision, 'saved_revision': project.saved_revision,
            'events': copy.deepcopy(project.events), 'modified_at': project.modified_at,
            'locked_frames': set(project.locked_frames), 'dirty_layers': dict(project.dirty_layers),
            'undo': [id(snapshot) for snapshot in project.undo_stack],
            'redo': [id(snapshot) for snapshot in project.redo_stack],
            'manifest': (project.directory / 'project.json').read_bytes(),
            'chunks': self.chunks(), 'last_save': copy.deepcopy(project.last_save),
            'reads': project.prior_store.reads,
            'contexts': {key: id(context) for key, context in project.contexts.items()},
        }
        # Inject failure at the real atomic publication boundary after the
        # temporary JSON has been written, as with a read-only/full filesystem.
        with patch('server.domain.os.replace', side_effect=OSError('Injected metadata publication failure')):
            with patch.object(self.client._transport, 'raise_server_exceptions', False):
                response = self.client.post(self.prefix + '/frame-locks',
                    json={'revision': project.revision, 'times': [0], 'locked': locked})
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(project.revision, before['revision'])
        self.assertEqual(project.saved_revision, before['saved_revision'])
        self.assertEqual(project.events, before['events'])
        self.assertEqual(project.modified_at, before['modified_at'])
        self.assertEqual(project.locked_frames, before['locked_frames'])
        self.assertEqual(project.dirty_layers, before['dirty_layers'])
        self.assertEqual([id(snapshot) for snapshot in project.undo_stack], before['undo'])
        self.assertEqual([id(snapshot) for snapshot in project.redo_stack], before['redo'])
        self.assertEqual((project.directory / 'project.json').read_bytes(), before['manifest'])
        self.assertEqual(self.chunks(), before['chunks'])
        self.assertEqual(project.last_save, before['last_save'])
        self.assertEqual(project.prior_store.reads, before['reads'])
        self.assertEqual({key: id(context) for key, context in project.contexts.items()}, before['contexts'])
        self.assert_masks(expected)

    def test_failed_lock_publication_preserves_exact_clean_and_dirty_state(self):
        self.infer(t=1)
        self.post('save')
        self.assert_failed_frame_change_is_atomic(locked=True)
        self.brush(t=1, radius=0)
        self.assert_failed_frame_change_is_atomic(locked=True)
        self.assertEqual(self.project.locked_frames, set())

    def test_failed_unlock_publication_never_removes_clean_or_dirty_protection(self):
        self.infer(t=1)
        self.post('save')
        self.lock()
        self.assert_failed_frame_change_is_atomic(locked=False)
        self.brush(t=1, radius=0)
        self.assert_failed_frame_change_is_atomic(locked=False)
        result = self.client.post(self.prefix + '/segment-all-layers',
                                 json={'revision': self.project.revision, 'time': 0})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.project.locked_frames, {0})

    def test_locked_empty_frame_rejects_every_direct_annotation_operation(self):
        self.lock()
        before = self.masks()
        revision, history, contexts = self.project.revision, len(self.project.undo_stack), dict(self.project.contexts)
        actions = [('brush', dict(time=0, slice=1, organ=label, points=[[2, 2]], radius=0)) for label in (1, 2, 3)]
        actions += [(action, dict(time=0, slice=1, organ=2)) for action in ('clear-slice', 'clear-points')]
        actions += [('inference', dict(time=0, slice=1, organ=2, mode='organ')),
                    ('inference', dict(time=0, slice=1, organ=2, mode='click', point={'x': 2, 'y': 2, 'positive': False})),
                    ('segment-all', dict(time=0, slice=1)), ('segment-all-layers', dict(time=0)),
                    ('clear-all-masks-slice', dict(time=0, slice=1)),
                    ('resolve-overlaps', dict(time=0, slice=1, priority=2))]
        for action, request in actions:
            result = self.client.post(self.prefix + '/' + action, json={'revision': revision, **request})
            self.assertEqual(result.status_code, 400, result.text)
            self.assertIn('frame', result.text.lower())
        self.assert_masks(before)
        self.assertEqual(self.project.contexts, contexts)
        self.assertEqual(self.project.revision, revision)
        self.assertEqual(len(self.project.undo_stack), history)
        self.assertEqual(self.predictor.calls, [])

    def test_all_frames_skips_locked_targets_and_scoped_undo_remains_available(self):
        self.infer(label=2, t=0, z=2)
        self.post('save')
        before = self.masks()
        contexts = {key: value for key, value in self.project.contexts.items() if key[0] == 0}
        self.lock()
        self.predictor.calls.clear()
        job = self.wait_job(self.post('segment-all-time-frames')['id'])
        self.assertEqual(job['state'], 'completed', job)
        self.assertEqual(job['target_times'], [1])
        self.assertEqual(job['skipped_locked_frames'], [0])
        self.assertEqual(job['total_frames'], 1)
        self.assertEqual(job['total_slices'], 3)
        self.assertEqual(job['completed_slices'], 3)
        self.assertEqual(len(self.predictor.calls), 9)
        for index, call in enumerate(self.predictor.calls):
            np.testing.assert_array_equal(call['image'], self.project.voxels[1, index // 3])
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0], before[label][0])
        for key, context in contexts.items():
            self.assertIs(self.project.contexts[key], context)
        self.assertEqual({key[0] for key in self.project.undo_stack[-1].planes}, {1})
        after = self.masks()
        self.post('undo')
        self.assert_masks(before)
        self.post('redo')
        self.assert_masks(after)
        self.assertEqual(self.project.locked_frames, {0})
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_multi_gpu_dispatch_never_enqueues_locked_frame(self):
        self.lock()
        counters = self.fake_pool()
        job = self.wait_job(self.post('segment-all-time-frames', gpu_ids=[4, 5])['id'])
        self.assertEqual(job['state'], 'completed', job)
        self.assertEqual(job['completed_slices'], 3)
        self.assertEqual(len(counters['calls']), 3)
        self.assertTrue(all(call.rsplit(':', 2)[1] == '1' for call in counters['calls']))
        self.assertTrue(all(not mask[0].any() for mask in self.project.masks.values()))

    def test_clear_layer_from_locked_current_frame_only_changes_unlocked_times(self):
        self.brush(t=0)
        self.brush(t=1)
        self.post('save')
        before = self.masks()
        contexts = dict(self.project.contexts)
        self.lock()
        self.post('clear-all-masks-layer', time=0, slice=1)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0], before[label][0])
            self.assertFalse(self.project.masks[label][1, 1].any())
        self.assertIs(self.project.contexts[(0, 1, 1)], contexts[(0, 1, 1)])
        self.assertEqual({key[0] for key in self.project.undo_stack[-1].planes}, {1})
        self.post('undo')
        self.assert_masks(before)
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_import_and_paired_baseline_skip_locked_frame_including_contexts(self):
        self.infer(label=2, t=0, z=2)
        self.post('save')
        before = self.masks()
        contexts = dict(self.project.contexts)
        self.lock()
        labels, _, _ = self.paired_fixture()
        image = sitk.GetImageFromArray(labels, isVector=False)
        image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image).status_code, 200)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0], before[label][0])
            np.testing.assert_array_equal(self.project.masks[label][1], labels[1] == label)
        self.assertIs(self.project.contexts[(0, 2, 2)], contexts[(0, 2, 2)])
        self.post('undo')
        self.assert_masks(before)
        with patch('server.app.ROOT', self.root):
            self.post('paired-annotation')
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][0], before[label][0])
            np.testing.assert_array_equal(self.project.masks[label][1], labels[1] == label)
        self.assertIs(self.project.contexts[(0, 2, 2)]['prior'], contexts[(0, 2, 2)]['prior'])
        self.post('undo')
        self.assert_masks(before)
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_undo_redo_cannot_restore_into_a_locked_frame_and_stack_is_unchanged(self):
        self.brush(t=0)
        before = self.masks()
        self.lock()
        stack = len(self.project.undo_stack)
        response = self.client.post(self.prefix + '/undo', json={'revision': self.project.revision})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.project.undo_stack), stack)
        self.assert_masks(before)
        self.lock(locked=False)
        self.post('undo')
        after = self.masks()
        self.lock()
        response = self.client.post(self.prefix + '/redo', json={'revision': self.project.revision})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.project.redo_stack), 1)
        self.assert_masks(after)
        self.lock(locked=False)
        self.post('redo')
        self.assert_masks(before)

    def test_lock_validation_stale_requests_and_active_jobs_are_nonmutating(self):
        for times in ([], [0, 0], [2], [-1], [True], ['0'], [0.0]):
            response = self.client.post(self.prefix + '/frame-locks',
                                        json={'revision': self.project.revision, 'times': times, 'locked': True})
            self.assertIn(response.status_code, (400, 422), response.text)
        self.assertEqual(self.project.locked_frames, set())
        self.lock()
        response = self.client.post(self.prefix + '/frame-locks', json={'revision': 0, 'times': [1], 'locked': True})
        self.assertEqual(response.status_code, 400)
        self.predictor.block = True
        job = self.infer(t=1, wait=False)
        self.assertTrue(self.predictor.entered.wait(5))
        response = self.client.post(self.prefix + '/frame-locks',
                                    json={'revision': self.project.revision, 'times': [1], 'locked': True})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.project.locked_frames, {0})
        self.predictor.release.set()
        self.assertEqual(self.wait_job(job['id'])['state'], 'completed')

    def test_final_writeback_guard_discards_results_if_protection_changes_outside_api(self):
        self.predictor.block = True
        job = self.infer(t=1, wait=False)
        self.assertTrue(self.predictor.entered.wait(5))
        before = self.masks()
        with self.project.lock:
            self.project.locked_slices.add((1, 1))
        with self.assertLogs('server.inference', level='ERROR') as logged:
            self.predictor.release.set()
            self.assertEqual(self.wait_job(job['id'])['state'], 'failed')
        self.assertEqual([(record.getMessage(), str(record.exc_info[1])) for record in logged.records],
                         [('Inference task failed', 'Time frame 2, layer 2 is locked. '
                           'Unlock this slice before changing its annotations.')])
        self.assert_masks(before)
        self.assertEqual(self.project.undo_stack, [])

    def test_all_locked_bulk_jobs_and_import_are_rejected_without_changes(self):
        self.lock((0, 1))
        before = self.masks()
        for action, details in [('segment-all-time-frames', {}), ('clear-all-masks-layer', {'time': 0, 'slice': 1})]:
            result = self.client.post(self.prefix + '/' + action, json={'revision': self.project.revision, **details})
            self.assertEqual(result.status_code, 400)
        image = sitk.GetImageFromArray(np.ones(self.project.voxels.shape, np.uint8), isVector=False)
        image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image).status_code, 400)
        self.assert_masks(before)
        self.assertEqual(self.predictor.calls, [])

    def test_rejected_input_preparation_never_leaves_an_orphan_pending_job(self):
        with patch.object(self.app.state.jobs, '_plane', side_effect=ValueError('Invalid input preparation')):
            for action, details in [('inference', {'time': 0, 'slice': 1, 'organ': 2, 'mode': 'organ'}),
                                    ('segment-all', {'time': 0, 'slice': 1})]:
                result = self.client.post(self.prefix + '/' + action, json={'revision': self.project.revision, **details})
                self.assertEqual(result.status_code, 400, result.text)
                self.assertIsNone(self.project.pending_job)
                self.assertEqual(self.app.state.jobs.active_tasks, 0)
                self.assertEqual(self.project.undo_stack, [])
        self.assertEqual(self.infer(mode='organ')['state'], 'completed')


class PropagationFrameLockTests(unittest.TestCase):
    setUp = temporal.TemporalPropagationTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    wait_job = fixtures.PlatformTests.wait_job
    reference_predictor = temporal.TemporalPropagationTests.reference_predictor
    request = temporal.TemporalPropagationTests.request
    propagate = temporal.TemporalPropagationTests.propagate
    copy_masks = temporal.TemporalPropagationTests.copy_masks

    def test_locked_reference_is_usable_but_locked_target_is_skipped_even_with_overwrite(self):
        self.post('frame-locks', times=[0, 1], locked=True)
        before = self.copy_masks()
        preview = self.post('propagation-preview', **self.request(overwrite_edited=True))
        self.assertEqual(preview['skipped_locked_frames'], [1])
        self.assertEqual([entry['time'] for entry in preview['eligible']], [3, 5])
        self.assertIn({'time': 1, 'organ': 2, 'reason': 'slice_locked'}, preview['skipped'])
        result = self.propagate(overwrite_edited=True)
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['target_times'], [3, 5])
        self.assertEqual(result['total_slices'], 2)
        self.assertEqual(result['skipped_locked_frames'], [1])
        self.assertEqual(len(self.reference_calls), 6)
        for call in self.reference_calls:
            self.assertEqual(len(call['references']), 3)
            np.testing.assert_array_equal(call['references'][0][0], self.project.voxels[0, 1])
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label][[0, 1, 2, 4]], before[label][[0, 1, 2, 4]])
        self.post('undo')
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])


if __name__ == '__main__':
    unittest.main()
