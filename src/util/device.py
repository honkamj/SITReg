"""Device handling related utils"""

from torch import device as torch_device
from torch.cuda import get_device_name as get_cuda_device_name


def get_device_name(device: torch_device) -> str:
    """Get device name"""
    if device.type == 'cpu':
        return 'cpu'
    if device.type == 'cuda':
        return get_cuda_device_name(device)
    raise ValueError('Unknown device type')
