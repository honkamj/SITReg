"""Common interfaces for algorithms"""

from abc import ABC, abstractmethod

from torch import Tensor


class IInterpolator(ABC):
    """Interpolates/extrapolates values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        """Interpolate

        Args:
            volume: Tensor with shape (batch_size, *channel_dims, dim_1, ..., dim_{n_dims})
            coordinates: Tensor with shape (batch_size, n_dims, *target_shape)

        Returns: Tensor with shape (batch_size, *channel_dims, *target_shape)
        """
