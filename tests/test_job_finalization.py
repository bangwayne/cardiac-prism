"""Terminal inference results must release the study before clients can see them."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

import test_platform as fixtures


class JobFinalizationTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job

    def submit_mode(self, mode):
        if mode == 'single':
            return self.infer(mode='organ', wait=False)
        if mode == 'all':
            return self.post('segment-all', time=0, slice=1)
        if mode == 'layers':
            return self.post('segment-all-layers', time=0)
        if mode == 'propagation':
            self.predictor.predict_reference = lambda image, references, label: self.predictor(
                image, [], label, 'organ', None, 0)
            return self.post('propagate-time', slice=1, reference_times=[0], target_times=[1],
                             organs=[1], overwrite_edited=True)
        return self.post('segment-all-time-frames')

    def check_terminal_publication(self, mode, outcome):
        manager = self.app.state.jobs
        self.brush()
        self.predictor.entered.clear()
        self.predictor.release.clear()
        self.predictor.block = True
        self.predictor.fail = outcome == 'failed'
        clearing = threading.Event()
        release_cleanup = threading.Event()
        project = self.project
        original_setattr = type(project).__setattr__

        def pause_reservation_release(instance, name, value):
            if instance is project and name == 'pending_job' and value is None:
                clearing.set()
                if not release_cleanup.wait(5):
                    raise RuntimeError('Reservation cleanup test barrier timed out')
            original_setattr(instance, name, value)

        # Pause the actual reservation release, after inference and any commit.
        # Previously a completed/failed/stale status was already observable at
        # this point, although the next edit still failed as "inference active".
        with patch.object(type(project), '__setattr__', pause_reservation_release):
            try:
                job = self.submit_mode(mode)
                self.assertTrue(self.predictor.entered.wait(5))
                if outcome == 'stale':
                    with project.lock:
                        project.revision += 1
                self.predictor.release.set()
                self.assertTrue(clearing.wait(5), 'Worker did not reach reservation cleanup')
                # The worker is held at the event barrier, so these values
                # cannot change until cleanup is released.
                self.assertEqual(manager.jobs[job['id']]['state'], 'running')
                self.assertNotIn('finished_at', manager.jobs[job['id']])
                self.assertEqual(manager.active_tasks, 1)
                self.assertEqual(project.pending_job, job['id'])
            finally:
                self.predictor.release.set()
                release_cleanup.set()

            result = self.wait_job(job['id'])
            self.assertEqual(result['state'], outcome, result)
            self.assertIn('finished_at', result)
            self.assertGreaterEqual(result['seconds'], 0)
            if 'phase' in result:
                self.assertEqual(result['phase'], outcome)
                self.assertEqual(result['phase_started_at'], result['finished_at'])
            self.assertIsNone(project.pending_job)
            self.assertEqual(manager.active_tasks, 0)
            # No separate wait for a finally block: observing the terminal
            # status alone must suffice for Undo, editing and another request.
            self.post('undo')
            self.brush(points=[[10, 8]], radius=0)
            self.predictor.block = False
            self.predictor.fail = False
            following = self.infer(mode='organ')
            self.assertEqual(following['state'], 'completed', following)
            self.assertEqual(manager.active_tasks, 0)

    def test_single_slice_completed_failed_and_stale_are_ready_for_immediate_edit(self):
        for outcome in ('completed', 'failed', 'stale'):
            with self.subTest(outcome=outcome):
                self.check_terminal_publication('single', outcome)

    def test_all_organs_completed_failed_and_stale_are_ready_for_immediate_edit(self):
        for outcome in ('completed', 'failed', 'stale'):
            with self.subTest(outcome=outcome):
                self.check_terminal_publication('all', outcome)

    def test_all_layers_completed_failed_and_stale_are_ready_for_immediate_edit(self):
        for outcome in ('completed', 'failed', 'stale'):
            with self.subTest(outcome=outcome):
                self.check_terminal_publication('layers', outcome)

    def test_all_time_frames_completed_failed_and_stale_are_ready_for_immediate_edit(self):
        for outcome in ('completed', 'failed', 'stale'):
            with self.subTest(outcome=outcome):
                self.check_terminal_publication('frames', outcome)

    def test_propagation_completed_failed_and_stale_are_ready_for_immediate_edit(self):
        for outcome in ('completed', 'failed', 'stale'):
            with self.subTest(outcome=outcome):
                self.check_terminal_publication('propagation', outcome)

    def test_cancelled_worker_cannot_release_a_newer_jobs_reservation(self):
        manager = self.app.state.jobs
        first_entered, first_release = threading.Event(), threading.Event()
        second_entered, second_release = threading.Event(), threading.Event()
        predictor = self.predictor
        calls = []

        def gated_prediction(*args):
            calls.append(1)
            entered, release = ((first_entered, first_release) if len(calls) == 1
                                else (second_entered, second_release))
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Prediction test barrier timed out')
            return predictor(*args)

        manager.model.predictor = gated_prediction
        try:
            first = self.infer(mode='organ', wait=False)
            self.assertTrue(first_entered.wait(5))
            response = self.client.post('/api/jobs/' + first['id'] + '/cancel')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['state'], 'cancelled')
            self.assertNotIn('finished_at', response.json())
            self.assertEqual(manager.active_tasks, 1)
            with self.assertRaisesRegex(ValueError, 'cancelled tasks still computing'):
                manager.set_device('cpu')
            # Cancellation promises no further commit and intentionally allows
            # refinement while the old model call finishes in the background.
            self.brush(points=[[10, 8]], radius=0)
            second = self.infer(mode='organ', wait=False)
            self.assertEqual(manager.active_tasks, 2)
            first_release.set()
            self.assertTrue(second_entered.wait(5))
            self.assertEqual(self.project.pending_job, second['id'])
            self.assertEqual(manager.active_tasks, 1)
            first_result = manager.get(first['id'])
            self.assertEqual(first_result['state'], 'cancelled')
            self.assertIn('finished_at', first_result)
            self.assertEqual(len(self.project.undo_stack), 1)
            self.assertEqual(self.project.masks[1][0, 1, 8, 10], 1)
            response = self.client.post(self.prefix + '/undo', json={'revision': self.project.revision})
            self.assertEqual(response.status_code, 400)
            second_release.set()
            self.assertEqual(self.wait_job(second['id'])['state'], 'completed')
            self.assertEqual(manager.active_tasks, 0)
            self.assertIsNone(self.project.pending_job)
        finally:
            first_release.set()
            second_release.set()

    def test_http_cancellation_releases_reservation_before_status_is_observable(self):
        manager = self.app.state.jobs
        self.predictor.block = True
        clearing, release_cleanup = threading.Event(), threading.Event()
        project = self.project
        original_setattr = type(project).__setattr__

        def pause_reservation_release(instance, name, value):
            if instance is project and name == 'pending_job' and value is None:
                clearing.set()
                if not release_cleanup.wait(5):
                    raise RuntimeError('Cancellation cleanup test barrier timed out')
            original_setattr(instance, name, value)

        first = self.infer(mode='organ', wait=False)
        self.assertTrue(self.predictor.entered.wait(5))
        with patch.object(type(project), '__setattr__', pause_reservation_release), ThreadPoolExecutor(1) as executor:
            try:
                cancelling = executor.submit(self.client.post, '/api/jobs/' + first['id'] + '/cancel')
                self.assertTrue(clearing.wait(5))
                # GET uses this lock. While the reservation is still present,
                # it must not see the cancellation's intermediate state.
                acquired = manager.lock.acquire(blocking=False)
                if acquired:
                    manager.lock.release()
                self.assertFalse(acquired, 'Cancellation exposed its status before releasing the study')
                self.assertEqual(project.pending_job, first['id'])
                self.assertEqual(manager.active_tasks, 1)
            finally:
                release_cleanup.set()
            response = cancelling.result(timeout=5)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['state'], 'cancelled')
            self.assertEqual(manager.get(first['id'])['state'], 'cancelled')
            self.assertIsNone(project.pending_job)
            self.brush(radius=0)
        self.predictor.release.set()
        manager.executor.submit(lambda: None).result(timeout=5)
        self.assertEqual(manager.active_tasks, 0)
        self.assertEqual(len(project.undo_stack), 1)

    def test_cancel_contending_with_successful_commit_cannot_relabel_it_cancelled(self):
        manager = self.app.state.jobs
        self.predictor.block = True
        clearing, release_cleanup, cancel_entered = threading.Event(), threading.Event(), threading.Event()
        project = self.project
        original_setattr = type(project).__setattr__
        original_get = manager.get

        def pause_reservation_release(instance, name, value):
            if instance is project and name == 'pending_job' and value is None:
                clearing.set()
                if not release_cleanup.wait(5):
                    raise RuntimeError('Commit cleanup test barrier timed out')
            original_setattr(instance, name, value)

        def observe_cancel_entry(job_id):
            cancel_entered.set()
            return original_get(job_id)

        with patch.object(type(project), '__setattr__', pause_reservation_release), ThreadPoolExecutor(1) as executor:
            try:
                job = self.infer(mode='organ', wait=False)
                self.assertTrue(self.predictor.entered.wait(5))
                self.predictor.release.set()
                self.assertTrue(clearing.wait(5))
                with patch.object(manager, 'get', observe_cancel_entry):
                    cancelling = executor.submit(self.client.post, '/api/jobs/' + job['id'] + '/cancel')
                    self.assertTrue(cancel_entered.wait(5))
                    acquired = manager.lock.acquire(blocking=False)
                    if acquired:
                        manager.lock.release()
                    self.assertFalse(acquired)
                    self.assertEqual(manager.jobs[job['id']]['state'], 'running')
                    release_cleanup.set()
                    response = cancelling.result(timeout=5)
            finally:
                self.predictor.release.set()
                release_cleanup.set()
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['state'], 'completed')
            self.assertEqual(manager.get(job['id'])['state'], 'completed')
            self.assertIsNone(project.pending_job)
            self.assertEqual(manager.active_tasks, 0)
            self.assertEqual(len(project.undo_stack), 1)
            self.assertTrue(project.masks[1].any())
            self.post('undo')


if __name__ == '__main__':
    unittest.main()
