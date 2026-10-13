"""CUDA inventory compatible with the release's pinned PyTorch version."""


def device_properties():
    import torch

    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    if count:
        return [torch.cuda.get_device_properties(index) for index in range(count)]

    # Torch 1.13's NVML parser mistakes CUDA_VISIBLE_DEVICES UUIDs for an
    # empty device list. Use the CUDA runtime for both count and properties;
    # the public property accessor repeats that same broken count check.
    torch.cuda.init()
    count = torch._C._cuda_getDeviceCount()
    return [torch.cuda._get_device_properties(index) for index in range(count)]
