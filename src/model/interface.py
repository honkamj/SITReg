"""Interfaces for models"""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.nn import Module


class INormalizerFactory(ABC):
    """Factory for generating normalization layers"""

    @abstractmethod
    def build(self, n_channels: int) -> Module:
        """Build normalization layer"""


class IActivationFactory(ABC):
    """Factory for generating activation layers"""

    @abstractmethod
    def build(self) -> Module:
        """Build activation layer"""


class FunctionalComponent(Module):
    """Functional neural network component (takes parameters as batched input)"""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of parameters that must be given as parameter vector"""

    def __call__(self, *args, parameters: Tensor | None = None, **kwargs: Any) -> Any:
        """Call the component with parameters"""
        return super().__call__(*args, **kwargs | {"parameters": parameters})
