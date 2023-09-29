"""Identity mapping"""

from torch import device as torch_device, dtype as torch_dtype
from .base import BaseComposableMapping
from .interface import IMaskedTensor


class ComposableIdentity(BaseComposableMapping):
    """Identity mapping"""
    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates

    def invert(self, **_inversion_parameters) -> 'ComposableIdentity':
        return ComposableIdentity()

    def detach(self) -> 'ComposableIdentity':
        return self

    def to_device(self, device: torch_device) -> 'ComposableIdentity':
        return self

    def to_dtype(self, dtype: torch_dtype) -> 'ComposableIdentity':
        return self
