"""Physical GPU inventory and isolated, lazy inference workers (spawn, never fork CUDA)."""
from __future__ import annotations

import csv
import multiprocessing
import os
import subprocess
import threading
import time


class GPUStartupCancelled(RuntimeError):
    """A model was cancelled before it became available for inference."""


def gpu_inventory():
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        try:
            index, uuid, name, total, free, usage = (value.strip() for value in row)
            available = int(free) >= int(os.environ.get('CARDIAC_PRISM_GPU_MIN_FREE_MIB', '8192'))
            devices.append({'id': int(index), 'uuid': uuid, 'name': name,
                            'total_mib': int(total), 'free_mib': int(free), 'utilization': int(usage),
                            'available': available, 'reason': None if available else 'Insufficient free GPU memory'})
        except (ValueError, TypeError):
            continue
    return devices


def _worker(connection, physical_id):
    # These must be set before importing Torch in the child process.
    # A container can expose physical GPU 3 as CUDA device 0. UUIDs retain
    # their identity across Docker visibility and CUDA ordinal remapping.
    devices = {device['id']: device['uuid'] for device in gpu_inventory()}
    if physical_id not in devices:
        connection.send({'error': f'GPU {physical_id} is no longer visible.'})
        connection.close()
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[physical_id]
    os.environ['CARDIAC_PRISM_DEVICE'] = 'cuda:0'
    from .inference import ModelService
    try:
        model = ModelService()
        model.ensure_loaded()
        connection.send({'ready': True, 'checkpoint_sha256': model.checkpoint_sha256})
        while True:
            request = connection.recv()
            if request is None:
                break
            task_id, image, labels, *extra = request
            try:
                results = ({label: model.predict_reference(image, extra[0][label], label) for label in labels}
                           if extra else {label: model.predict(image, [], label, 'organ', None, 0) for label in labels})
                connection.send({'task_id': task_id, 'results': results})
            except Exception as error:
                connection.send({'task_id': task_id, 'error': str(error)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send({'error': str(error)})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class GPUWorker:
    def __init__(self, physical_id, cancelled=None):
        self.id = physical_id
        self.lock = threading.Lock()
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, physical_id),
                                       name=f'kprism-gpu-{physical_id}', daemon=True)
        try:
            self.process.start(); child.close()
            message = self._receive(180, cancelled=cancelled)
            if not message.get('ready'):
                raise RuntimeError(message.get('error', 'GPU worker did not initialize.'))
            self.checkpoint_sha256 = message['checkpoint_sha256']
        except Exception:
            child.close()
            self.close()
            raise

    def _receive(self, timeout, cancelled=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                raise GPUStartupCancelled(f'GPU {self.id} startup was cancelled.')
            if self.connection.poll(.2):
                return self.connection.recv()
            if not self.process.is_alive():
                raise RuntimeError(f'GPU {self.id} worker exited before returning a result.')
        self.close()
        raise RuntimeError(f'GPU {self.id} worker timed out.')

    def predict(self, task_id, image, labels):
        return self._predict((task_id, image, labels))

    def predict_reference(self, task_id, image, labels, references):
        return self._predict((task_id, image, labels, {label: references[label] for label in labels}))

    def _predict(self, request):
        task_id = request[0]
        with self.lock:
            self.connection.send(request)
            response = self._receive(120)
            if response.get('task_id') != task_id:
                raise RuntimeError('GPU worker returned a result for a different task.')
            if 'error' in response:
                raise RuntimeError(f'GPU {self.id}: {response["error"]}')
            return response['results']

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (EOFError, BrokenPipeError, OSError):
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate(); self.process.join(timeout=3)
        self.connection.close()


class GPUWorkerPool:
    def __init__(self):
        self.workers = {}
        self.lock = threading.Lock()
        self.creation_locks = {}
        self.closed = False

    def validate(self, ids):
        if not ids or len(ids) > 16 or len(set(ids)) != len(ids):
            raise ValueError('Select one or more distinct physical GPU IDs.')
        available = {device['id']: device for device in gpu_inventory()}
        for physical_id in ids:
            if physical_id not in available:
                raise ValueError(f'GPU {physical_id} is not present on this server.')
            if not available[physical_id]['available']:
                raise ValueError(f'GPU {physical_id} has insufficient free memory. Choose another GPU or use the current device.')

    def get(self, physical_id, cancelled=None):
        # Loading a checkpoint and starting CUDA can take several seconds. A
        # per-device lock prevents duplicate models without making unrelated
        # GPUs wait for one another's initialization.
        with self.lock:
            if self.closed:
                raise RuntimeError('GPU worker pool is closed.')
            creation_lock = self.creation_locks.setdefault(physical_id, threading.Lock())
        while not creation_lock.acquire(timeout=.1):
            if cancelled is not None and cancelled():
                raise GPUStartupCancelled(f'GPU {physical_id} startup was cancelled.')
        try:
            if cancelled is not None and cancelled():
                raise GPUStartupCancelled(f'GPU {physical_id} startup was cancelled.')
            with self.lock:
                if self.closed:
                    raise RuntimeError('GPU worker pool is closed.')
                worker = self.workers.get(physical_id)
                if worker is not None and not worker.process.is_alive():
                    self.workers.pop(physical_id)
            if worker is not None and not worker.process.is_alive():
                worker.close()
                worker = None
            if worker is not None:
                return worker
            # Recheck memory immediately before allocating a model.
            self.validate([physical_id])
            if cancelled is not None and cancelled():
                raise GPUStartupCancelled(f'GPU {physical_id} startup was cancelled.')
            worker = GPUWorker(physical_id, cancelled=cancelled)
            with self.lock:
                publish = not self.closed
                if publish:
                    self.workers[physical_id] = worker
            if not publish:
                worker.close()
                raise RuntimeError('GPU worker pool was closed during startup.')
            return worker
        finally:
            creation_lock.release()

    def get_for_job(self, physical_id, cancelled):
        return self.get(physical_id, cancelled=cancelled)

    def close(self):
        with self.lock:
            self.closed = True
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.close()
