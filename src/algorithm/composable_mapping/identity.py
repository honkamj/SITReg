"""Identity mapping"""

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
