"""Whole-cine transactions and concurrent physical-GPU scheduling, without CUDA."""
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_platform as fixtures
from server.domain import ProjectStore, require_same_geometry
from server.gpu_pool import GPUStartupCancelled, GPUWorkerPool


class AllTimeFramesTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job
    probability_predictor = fixtures.PlatformTests.probability_predictor

    def run_all(self, **details):
        return self.wait_job(self.post('segment-all-time-frames', **details)['id'])

    def drain(self):
        deadline = time.monotonic() + 5
        while self.app.state.jobs.active_tasks and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.app.state.jobs.active_tasks, 0)

    def test_each_time_and_slice_uses_native_image_without_shared_prompts(self):
        self.infer(label=1, t=1, z=2)
        self.predictor.calls.clear()
        result = self.run_all()
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['total_slices'], 6)
        self.assertEqual(result['completed_slices'], 6)
        self.assertEqual(result['completed_frames'], 2)
        self.assertEqual(len(self.predictor.calls), 18)
        for index, call in enumerate(self.predictor.calls):
            t, z = divmod(index // 3, 3)
            np.testing.assert_array_equal(call['image'], self.project.voxels[t, z])
            self.assertEqual(call['label'], index % 3 + 1)
            self.assertEqual(call['points'], [])
            self.assertIsNone(call['prior'])
            self.assertEqual(call['iteration'], 0)
            self.assertEqual(call['mode'], 'organ')
        self.assertEqual(len(self.project.contexts), 18)
        self.assertEqual(self.project.overlaps(), 0)

    def test_probability_competition_full_undo_redo_and_4d_export(self):
        self.infer(label=3, t=0, z=1)
        self.infer(label=1, t=1, z=2)
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        old_prior = self.project.contexts[(1, 2, 1)]['prior'].copy()
        revision = self.project.revision
        self.probability_predictor()
        self.assertEqual(self.run_all()['state'], 'completed')
        self.assertEqual(self.project.revision, revision + 1)
        for t in range(2):
            for z in range(3):
                self.assertEqual(self.project.masks[2][t, z, 5, 6], 1)
                self.assertEqual(self.project.masks[3][t, z, 2, 2], 1)
                self.assertEqual(self.project.masks[1][t, z, 1, 1], 1)
                for label in (1, 2, 3):
                    self.assertEqual(self.project.masks[label][t, z, 8, 11], 0)
                    self.assertEqual(self.project.contexts[(t, z, label)]['points'], [])
        after = {label: mask.copy() for label, mask in self.project.masks.items()}
        self.post('undo')
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])
        np.testing.assert_array_equal(self.project.contexts[(1, 2, 1)]['prior'], old_prior)
        self.assertEqual(len(self.project.contexts[(1, 2, 1)]['points']), 1)
        self.assertNotIn((0, 0, 2), self.project.contexts)
        self.post('redo')
        for label in after:
            np.testing.assert_array_equal(self.project.masks[label], after[label])
        self.post('save')
        reopened = ProjectStore(self.root / 'work/projects').get(self.id)
        for label in after:
            np.testing.assert_array_equal(reopened.masks[label], after[label])
        exported = self.root / 'all-frames.nii.gz'
        exported.write_bytes(self.client.get(self.prefix + '/export?kind=labelmap').content)
        image = sitk.ReadImage(str(exported))
        require_same_geometry(self.project.image, image)
        expected = sum(label * mask for label, mask in after.items())
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)

    def test_locks_protect_pixels_and_allow_scoped_undo_across_time(self):
        for t in range(2):
            for z in range(3):
                self.brush(label=1, t=t, z=z)
        self.post('lock', organ=1, locked=True)
        protected = self.project.masks[1].copy()
        self.probability_predictor()
        result = self.run_all()
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['protected_voxels'], 6)
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertEqual(self.project.overlaps(), 0)
        self.post('undo')
        np.testing.assert_array_equal(self.project.masks[1], protected)
        self.assertFalse(self.project.masks[2].any())
        self.assertFalse(self.project.masks[3].any())

    def test_failure_in_second_frame_does_not_publish_first_frame(self):
        self.brush(label=3, t=1, z=2)
        before = {label: mask.copy() for label, mask in self.project.masks.items()}
        revision = self.project.revision
        original = self.predictor.__call__
        def fail(image, *args):
            if len(self.predictor.calls) == 9:
                raise RuntimeError('Second time frame failed')
            return original(image, *args)
        self.app.state.model.predictor = fail
        result = self.run_all()
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['completed_frames'], 1)
        self.assertEqual(self.project.revision, revision)
        for label in before:
            np.testing.assert_array_equal(self.project.masks[label], before[label])

    def test_cancel_after_first_frame_is_atomic_and_blocks_device_switch_until_drained(self):
        entered = threading.Event()
        original = self.predictor.__call__
        def block(image, *args):
            if len(self.predictor.calls) == 9:
                entered.set()
                self.assertTrue(self.predictor.release.wait(5))
            return original(image, *args)
        self.app.state.model.predictor = block
        job = self.post('segment-all-time-frames')
        self.assertTrue(entered.wait(5))
        self.assertEqual(self.client.get('/api/jobs/' + job['id']).json()['completed_frames'], 1)
        self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status()
        self.assertEqual(self.client.post('/api/model/device', json={'device': 'cpu'}).status_code, 400)
        self.predictor.release.set(); self.drain()
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_stale_revision_discards_the_entire_cine(self):
        self.predictor.block = True
        job = self.post('segment-all-time-frames')
        self.assertTrue(self.predictor.entered.wait(5))
        with self.project.lock:
            self.project.revision += 1
        self.predictor.release.set()
        self.assertEqual(self.wait_job(job['id'])['state'], 'stale')
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_invalid_gpu_ids_rejected_before_any_prediction(self):
        inventory = [{'id': 4, 'available': True}, {'id': 5, 'available': False}]
        with patch('server.gpu_pool.gpu_inventory', return_value=inventory):
            for ids in ([], [4, 4], [5], [99], [-1], ['4'], [True]):
                response = self.client.post(self.prefix + '/segment-all-time-frames', json={'revision': 0, 'gpu_ids': ids})
                self.assertIn(response.status_code, (400, 422), response.text)
        self.assertEqual(self.predictor.calls, [])
        self.assertIsNone(self.project.pending_job)

    def fake_pool(self, block=False, mismatch=False, fail=False):
        gate = threading.Barrier(2)
        release = self.predictor.release
        counters = {'active': 0, 'peak': 0, 'calls': [], 'entered': threading.Event()}
        guard = threading.Lock()
        predict = self.predictor
        class Worker:
            def __init__(worker, id):
                worker.id = id
                worker.checkpoint_sha256 = 'wrong' if mismatch and id == 5 else 'test-sha'
            def predict(worker, task_id, image, labels):
                with guard:
                    counters['active'] += 1
                    counters['peak'] = max(counters['peak'], counters['active'])
                    counters['calls'].append(task_id)
                try:
                    if len(counters['calls']) <= 2:
                        gate.wait(5)
                    counters['entered'].set()
                    if block:
                        if not release.wait(5):
                            raise RuntimeError('GPU test barrier timeout')
                    if fail and worker.id == 5:
                        raise RuntimeError('GPU worker test failure')
                    time.sleep(.01 if worker.id == 4 else .002)
                    return {label: predict(image, [], label, 'organ', None, 0) for label in labels}
                finally:
                    with guard:
                        counters['active'] -= 1
        class Pool:
            def validate(pool, ids):
                pass
            def get(pool, id):
                return Worker(id)
            def close(pool):
                pass
        self.app.state.jobs.gpu_pool.close()
        self.app.state.jobs.gpu_pool = Pool()
        return counters

    def test_two_gpu_scheduler_runs_concurrently_and_maps_out_of_order_results(self):
        counters = self.fake_pool()
        result = self.run_all(gpu_ids=[4, 5])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(counters['peak'], 2)
        self.assertEqual(result['completed_slices'], 6)
        self.assertEqual(sum(result['gpu_completed'].values()), 6)
        self.assertEqual(set(result['gpu_completed']), {'4', '5'})
        coordinates = {(int(call.rsplit(':', 2)[1]), int(call.rsplit(':', 1)[1])) for call in counters['calls']}
        self.assertEqual(coordinates, {(t, z) for t in range(2) for z in range(3)})
        self.assertEqual(self.project.overlaps(), 0)
        for t in range(2):
            for z in range(3):
                self.assertEqual(int(self.project.masks[1][t, z].sum()), 20)

    def test_parallel_startup_reports_ready_cards_before_prediction_and_one_atomic_commit(self):
        gate = threading.Barrier(2)
        release_startup = threading.Event()
        self.addCleanup(release_startup.set)
        predict = self.predictor
        predict.block = True

        class Worker:
            checkpoint_sha256 = 'test-sha'
            def predict(worker, task_id, image, labels):
                return {label: predict(image, [], label, 'organ', None, 0) for label in labels}

        class Pool:
            def validate(pool, ids):
                pass
            def get(pool, gpu_id):
                gate.wait(3)
                if gpu_id == 5 and not release_startup.wait(5):
                    raise RuntimeError('Startup test barrier timed out')
                return Worker()
            def close(pool):
                pass

        self.app.state.jobs.gpu_pool = Pool()
        before_undo = len(self.project.undo_stack)
        job = self.post('segment-all-time-frames', gpu_ids=[4, 5])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = self.client.get('/api/jobs/' + job['id']).json()
            if current['ready_gpu_ids'] == [4]:
                break
            time.sleep(.005)
        self.assertEqual(current['phase'], 'preparing', current)
        self.assertEqual(current['ready_gpu_ids'], [4], current)
        self.assertEqual(current['preparing_gpu_ids'], [5])
        self.assertEqual(current['execution_devices'], ['GPU 4'])
        self.assertEqual(current['completed_slices'], 0)
        self.assertEqual(predict.calls, [])
        # A poll response must remain a snapshot while the live job changes.
        release_startup.set()
        self.assertTrue(predict.entered.wait(3))
        computing = self.client.get('/api/jobs/' + job['id']).json()
        self.assertEqual(computing['phase'], 'predicting')
        self.assertEqual(computing['ready_gpu_ids'], [4, 5])
        self.assertEqual(computing['preparing_gpu_ids'], [])
        self.assertEqual(current['ready_gpu_ids'], [4])
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))
        predict.release.set()
        result = self.wait_job(job['id'])
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(result['result_visibility'], 'on_completion')
        self.assertGreaterEqual(result['first_result_seconds'], result['gpu_startup_seconds'])
        self.assertEqual(sum(result['gpu_completed'].values()), 6)
        self.assertEqual(len(self.project.undo_stack), before_undo + 1)

    def test_cancel_during_concurrent_startup_stops_loads_and_never_predicts(self):
        entered = threading.Barrier(3)
        stopped = []

        class Pool:
            def validate(pool, ids):
                pass
            def get_for_job(pool, gpu_id, cancelled):
                entered.wait(3)
                deadline = time.monotonic() + 3
                while not cancelled() and time.monotonic() < deadline:
                    time.sleep(.005)
                if cancelled():
                    stopped.append(gpu_id)
                    raise GPUStartupCancelled('Cancelled test startup')
                raise RuntimeError('Cancellation was not passed to startup')
            def close(pool):
                pass

        self.app.state.jobs.gpu_pool = Pool()
        job = self.post('segment-all-time-frames', gpu_ids=[4, 5])
        entered.wait(3)
        current = self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status().json()
        self.assertEqual(current['phase'], 'cancelled')
        self.drain()
        result = self.client.get('/api/jobs/' + job['id']).json()
        self.assertEqual(result['state'], 'cancelled')
        self.assertEqual(result['preparing_gpu_ids'], [])
        self.assertEqual(set(stopped), {4, 5})
        self.assertEqual(self.predictor.calls, [])
        self.assertEqual(self.project.revision, 0)
        self.assertEqual(len(self.project.undo_stack), 0)
        self.assertIsNone(self.project.pending_job)

    def test_failed_gpu_startup_cancels_other_loads_without_publishing(self):
        gate = threading.Barrier(2)
        stopped = threading.Event()

        class Pool:
            def validate(pool, ids):
                pass
            def get_for_job(pool, gpu_id, cancelled):
                gate.wait(3)
                if gpu_id == 4:
                    raise RuntimeError('GPU 4 model loading failed')
                deadline = time.monotonic() + 3
                while not cancelled() and time.monotonic() < deadline:
                    time.sleep(.005)
                if cancelled():
                    stopped.set()
                    raise GPUStartupCancelled('Other startup failed')
                raise RuntimeError('Other model load was not stopped')
            def close(pool):
                pass

        self.app.state.jobs.gpu_pool = Pool()
        result = self.run_all(gpu_ids=[4, 5])
        self.assertEqual(result['state'], 'failed', result)
        self.assertEqual(result['phase'], 'failed')
        self.assertIn('GPU 4 model loading failed', result['error'])
        self.assertEqual(result['gpu_errors'], {'4': 'GPU 4 model loading failed'})
        self.assertEqual(result['preparing_gpu_ids'], [])
        self.assertTrue(stopped.is_set())
        self.assertEqual(self.predictor.calls, [])
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_gpu_cancellation_drains_workers_and_stops_new_dispatch(self):
        counters = self.fake_pool(block=True)
        job = self.post('segment-all-time-frames', gpu_ids=[4, 5])
        self.assertTrue(counters['entered'].wait(5))
        self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status()
        self.predictor.release.set(); self.drain()
        self.assertEqual(len(counters['calls']), 2)
        self.assertEqual(counters['active'], 0)
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_progress_response_remains_stable_as_gpu_counters_change(self):
        counters = self.fake_pool(block=True)
        job = self.post('segment-all-time-frames', gpu_ids=[4, 5])
        self.assertTrue(counters['entered'].wait(5))
        snapshot = self.app.state.jobs.get(job['id'])
        self.assertEqual(snapshot['gpu_completed'], {})
        self.predictor.release.set()
        self.assertEqual(self.wait_job(job['id'])['state'], 'completed')
        self.assertEqual(snapshot['gpu_completed'], {})

    def test_gpu_worker_failure_is_atomic(self):
        self.fake_pool(fail=True)
        result = self.run_all(gpu_ids=[4, 5])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(self.project.revision, 0)
        self.assertFalse(any(mask.any() for mask in self.project.masks.values()))

    def test_mismatched_gpu_checkpoints_are_rejected(self):
        counters = self.fake_pool(mismatch=True)
        result = self.run_all(gpu_ids=[4, 5])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(counters['calls'], [])
        self.assertEqual(self.project.revision, 0)

    def test_all_locked_or_overlapping_locked_structures_rejected(self):
        for label in (1, 2, 3):
            self.project.locks[label] = True
        response = self.client.post(self.prefix + '/segment-all-time-frames', json={'revision': 0})
        self.assertEqual(response.status_code, 400)
        self.project.locks[3] = False
        self.project.masks[1][1, 2, 5, 6] = 1
        self.project.masks[2][1, 2, 5, 6] = 1
        response = self.client.post(self.prefix + '/segment-all-time-frames', json={'revision': 0})
        self.assertEqual(response.status_code, 400)
