"""Common interfaces for algorithms"""

from abc import ABC, abstractmethod
from typing import Callable

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


class IFixedPointSolver(ABC):
    """Interface for fixed point solvers"""

    @abstractmethod
    def solve(
        self,
        fixed_point_function: Callable[[Tensor], Tensor],
        initial_value: Tensor,
    ) -> Tensor:
        """Solve fixed point problem

        Args:
            fixed_point_function: Function to be iterated
            initial_value: Initial iteration value

        Returns: Solution of the fixed point iteration
        """


class IFixedPointStopCriterion(ABC):
    """Defines stopping criterion for fixed point iteration"""

    @abstractmethod
    def should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        """Return whether iterating should be stopped at end of an iteration

        After initial guess iteration == 0
        """

    @abstractmethod
    def should_stop_before(self, iteration_to_start: int) -> bool:
        """Return whether iterating should be continued at beginning of an iteration

        After initial guess iteration == 0
        """
