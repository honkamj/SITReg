"""Normalizer implementations"""

from typing import Callable, Optional

from torch import Tensor
from torch.nn import GroupNorm

from model.interface import INormalizerFactory
from util.ndimensional_operators import instance_norm_nd


class GroupNormalizerFactory(INormalizerFactory):
    """Factory for generating group normalization layers"""

    def __init__(self, n_groups: int) -> None:
        self._n_groups = n_groups

    def build(self, n_channels: int) -> Callable[[Tensor], Tensor]:
        return GroupNorm(num_groups=self._n_groups, num_channels=n_channels)


class InstanceNormalizerFactory(INormalizerFactory):
    """Factory for generating instance normalization layers"""

    def __init__(self, n_dims: int) -> None:
        self._n_dims = n_dims

    def build(self, n_channels: int) -> Callable[[Tensor], Tensor]:
        return instance_norm_nd(self._n_dims)(num_features=n_channels)


class EmptyNormalizerFactory(INormalizerFactory):
    """Factory for generating placeholder normalization layers"""

    def build(self, n_channels: int) -> Callable[[Tensor], Tensor]:
        return lambda x: x


def get_normalizer_factory(normalizer_factory: Optional[INormalizerFactory]) -> INormalizerFactory:
    """Conver to empty normalizer factory if None"""
    if normalizer_factory is None:
        return EmptyNormalizerFactory()
    return normalizer_factory
