"""Interfaces for models"""

from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor


class INormalizerFactory(ABC):
    """Factory for generating normalization layers"""

    @abstractmethod
    def build(self, n_channels: int) -> Callable[[Tensor], Tensor]:
        """Build normalization layer"""
