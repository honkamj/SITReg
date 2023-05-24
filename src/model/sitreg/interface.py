"""SITReg interface"""

from abc import abstractmethod, ABC
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import Module


class IFeatureExtractor(Module, ABC):
    """Multi-resolution feature extractor for SITReg architecture"""

    @abstractmethod
    def get_shapes(self) -> Sequence[Sequence[int]]:
        """Get shapes at different levels, starting from largest shape

        Returns: Sequence of shapes of following form (n_features, dim_1_size, ..., dim_n_size)
        """

    @abstractmethod
    def _get_downsampling_factors(self) -> Sequence[Sequence[float]]:
        """Get downsampling factors at different levels, starting from smallest factor"""

    @abstractmethod
    def get_downsampling_factors(
        self, relative_to_downsampling_factors: Optional[Sequence[float]] = None
    ) -> Sequence[Sequence[float]]:
        """Get downsampling factors at different levels, starting from smallest factor

        Args:
            relative_to_downsampling_factors: Get downsampling factors
                relative to these factors
        """

    @abstractmethod
    def forward(self, image: Tensor) -> Sequence[Tensor]:
        """Compute features, starting from features with smallest downsampling factor

        Args:
            image: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})

        Returns:
            Sequence of Tensors with shapes (batch_size, n_features, dim_1, ..., dim_{n_dims})
        """
