"""Independent random streams and diagnostics for paired forecasting runs."""

from contextlib import contextmanager
import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    # Set before the first CUDA operation; strict mode fails on unsupported ops.
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(worker_id):
    """Top-level function so DataLoader workers can import it under spawn."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IsolatedTorchRNG:
    """Advancing auxiliary RNG stream that preserves the caller's RNG state.

    Runtime state is intentionally not a model parameter/buffer. Existing
    checkpoints support inference, not exact mid-training RNG restoration.
    """

    def __init__(self, seed):
        self.seed = seed
        self.cpu_state = torch.Generator().manual_seed(seed).get_state()
        self.cuda_states = {}

    @contextmanager
    def use(self, device):
        device = torch.device(device)
        devices = []
        if device.type == 'cuda':
            index = device.index if device.index is not None else torch.cuda.current_device()
            devices = [index]
            if index not in self.cuda_states:
                self.cuda_states[index] = torch.Generator(
                    device=torch.device('cuda', index)
                ).manual_seed(self.seed).get_state()
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu_state)
            for index in devices:
                torch.cuda.set_rng_state(self.cuda_states[index], index)
            try:
                yield
            finally:
                self.cpu_state = torch.get_rng_state()
                for index in devices:
                    self.cuda_states[index] = torch.cuda.get_rng_state(index)


def _fingerprint(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode('utf-8'))
        if tensor is None:
            digest.update(b'None')
            continue
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('ascii'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_fingerprint(*tensors):
    return _fingerprint((str(index), tensor) for index, tensor in enumerate(tensors))


def backbone_fingerprint(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return _fingerprint(
        (name, tensor) for name, tensor in sorted(model.state_dict().items())
        if not name.startswith('reverse_model.')
    )
