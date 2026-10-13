"""Concurrent GPU startup, reuse and cancellation without allocating CUDA."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from server.gpu_pool import GPUStartupCancelled, GPUWorker, GPUWorkerPool, _worker


class GPUStartupTests(unittest.TestCase):
    def test_uuid_visibility_uses_real_cuda_count_with_pinned_torch(self):
        import torch
        from server.inference import ModelService
        properties = SimpleNamespace(name='Example GPU', total_memory=24 * 1024 ** 3)
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-example'}), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=0), \
                patch.object(torch._C, '_cuda_getDeviceCount', return_value=1, create=True), \
                patch.object(torch.cuda, 'init'), \
                patch.object(torch.cuda, '_get_device_properties', return_value=properties, create=True), \
                patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('Invalid device id')):
            devices = ModelService().available_devices()
        self.assertEqual([device['id'] for device in devices], ['cpu', 'cuda:0'])

    def test_container_worker_uses_uuid_instead_of_remapped_cuda_ordinal(self):
        connection = Mock()
        connection.recv.return_value = None
        with patch('server.gpu_pool.gpu_inventory', return_value=[{'id': 3, 'uuid': 'GPU-example'}]), \
                patch('server.inference.ModelService') as model, \
                patch.dict('os.environ', {}, clear=False):
            import os
            _worker(connection, 3)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-example')
            self.assertEqual(os.environ['CARDIAC_PRISM_DEVICE'], 'cuda:0')
            model.return_value.ensure_loaded.assert_called_once()
        connection.close.assert_called_once()

    def test_removed_gpu_cannot_silently_use_a_different_device(self):
        connection = Mock()
        with patch('server.gpu_pool.gpu_inventory', return_value=[]):
            _worker(connection, 3)
        self.assertIn('no longer visible', connection.send.call_args.args[0]['error'])
        connection.close.assert_called_once()

    def pool(self):
        pool = GPUWorkerPool()
        pool.validate = Mock()
        self.addCleanup(pool.close)
        return pool

    @staticmethod
    def worker():
        worker = Mock()
        worker.process.is_alive.return_value = True
        return worker

    def test_different_gpu_models_initialize_concurrently(self):
        pool = self.pool()
        gate = threading.Barrier(2)
        created = {}

        def create(gpu_id, cancelled=None):
            gate.wait(3)  # Fails if a global pool lock serializes checkpoint loading.
            created[gpu_id] = self.worker()
            return created[gpu_id]

        with patch('server.gpu_pool.GPUWorker', side_effect=create), ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(pool.get, gpu_id) for gpu_id in (2, 3)]
            self.assertEqual([future.result(4) for future in futures], [created[2], created[3]])
        self.assertEqual(set(pool.workers), {2, 3})

    def test_same_gpu_simultaneous_requests_create_only_one_model(self):
        pool = self.pool()
        entered, release = threading.Event(), threading.Event()
        worker = self.worker()

        def create(gpu_id, cancelled=None):
            entered.set()
            self.assertTrue(release.wait(3))
            return worker

        with patch('server.gpu_pool.GPUWorker', side_effect=create) as constructor, ThreadPoolExecutor(2) as executor:
            first = executor.submit(pool.get, 2)
            self.assertTrue(entered.wait(3))
            second = executor.submit(pool.get, 2)
            release.set()
            self.assertIs(first.result(3), worker)
            self.assertIs(second.result(3), worker)
        self.assertEqual(constructor.call_count, 1)

    def test_cancelled_waiter_does_not_start_another_model(self):
        pool = self.pool()
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        worker = self.worker()

        def create(gpu_id, cancelled=None):
            entered.set()
            self.assertTrue(release.wait(3))
            return worker

        with patch('server.gpu_pool.GPUWorker', side_effect=create) as constructor, ThreadPoolExecutor(2) as executor:
            first = executor.submit(pool.get, 2)
            self.assertTrue(entered.wait(3))
            second = executor.submit(pool.get_for_job, 2, cancelled.is_set)
            cancelled.set()
            try:
                with self.assertRaises(GPUStartupCancelled):
                    second.result(3)
            finally:
                release.set()
            self.assertIs(first.result(3), worker)
        self.assertEqual(constructor.call_count, 1)

    def test_pool_close_during_startup_disposes_unpublished_worker(self):
        pool = self.pool()
        entered, release = threading.Event(), threading.Event()
        worker = self.worker()

        def create(gpu_id, cancelled=None):
            entered.set()
            self.assertTrue(release.wait(3))
            return worker

        with patch('server.gpu_pool.GPUWorker', side_effect=create), ThreadPoolExecutor(1) as executor:
            future = executor.submit(pool.get, 2)
            self.assertTrue(entered.wait(3))
            pool.close()
            release.set()
            with self.assertRaisesRegex(RuntimeError, 'closed during startup'):
                future.result(3)
        worker.close.assert_called_once()
        self.assertEqual(pool.workers, {})

    def test_failed_startup_can_be_retried_without_poisoning_the_pool(self):
        pool = self.pool()
        worker = self.worker()
        with patch('server.gpu_pool.GPUWorker', side_effect=[RuntimeError('model load failed'), worker]):
            with self.assertRaisesRegex(RuntimeError, 'model load failed'):
                pool.get(2)
            self.assertEqual(pool.workers, {})
            self.assertIs(pool.get(2), worker)

    def test_worker_startup_cancellation_terminates_child_and_closes_pipe(self):
        context = Mock()
        parent, child, process = Mock(), Mock(), Mock()
        context.Pipe.return_value = (parent, child)
        context.Process.return_value = process
        process.is_alive.return_value = True
        with patch('server.gpu_pool.multiprocessing.get_context', return_value=context):
            with self.assertRaises(GPUStartupCancelled):
                GPUWorker(2, cancelled=lambda: True)
        process.terminate.assert_called_once()
        parent.close.assert_called_once()
        child.close.assert_called()


if __name__ == '__main__':
    unittest.main()
