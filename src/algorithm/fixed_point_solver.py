"""Fixed point solvers"""

from abc import abstractmethod
from logging import getLogger
from typing import Callable, NamedTuple, Optional

from torch import Tensor
from torch import abs as torch_abs
from torch import bmm, empty, eye
from torch import max as torch_max
from torch import zeros, zeros_like
from torch.linalg import solve

from .interface import IFixedPointSolver, IFixedPointStopCriterion

logger = getLogger(__name__)


class BaseStopCriterion(IFixedPointStopCriterion):
    """Base stop criterion definining min and max number of iterations

    Args:
        min_iterations: Minimum number of iterations to use
        max_iterations: Maximum number of iterations to use
        check_convergence_every_nth_iteration: Check convergence criterion
            only every nth itheration, does not have effect on stopping based on
            min or max number of iterations.
    """

    def __init__(
        self,
        min_iterations: int = 2,
        max_iterations: int = 50,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        self._min_iterations = min_iterations
        self._max_iterations = max_iterations
        self._check_convergence_every_nth_iteration = check_convergence_every_nth_iteration

    def _should_check_convergence(self, iteration_to_end: int) -> bool:
        return ((iteration_to_end + 1) % self._check_convergence_every_nth_iteration) == 0

    @abstractmethod
    def _should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        """Return whether should stop after the iteration"""

    def should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        if iteration_to_end + 1 < self._min_iterations:
            return False
        return self._should_check_convergence(iteration_to_end) and self._should_stop_after(
            previous_iteration, current_iteration, iteration_to_end
        )

    def should_stop_before(self, iteration_to_start: int) -> bool:
        if self._max_iterations > iteration_to_start:
            return False
        return True


class FixedIterationCountStopCriterion(BaseStopCriterion):
    """Iteration is terminated based on fixed iteration count"""

    def __init__(
        self,
        n_iterations: int = 50,
    ) -> None:
        super().__init__(
            min_iterations=0, max_iterations=n_iterations, check_convergence_every_nth_iteration=1
        )

    def _should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        return False


class MaxElementWiseAbsStopCriterion(BaseStopCriterion):
    """Stops when no element-wise difference is larger than a threshold"""

    def __init__(
        self,
        min_iterations: int = 1,
        max_iterations: int = 50,
        max_error: float = 1e-2,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )
        self._max_error = max_error

    def _should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        max_difference = torch_max(torch_abs(previous_iteration - current_iteration))
        return bool(max_difference < self._max_error)


class RelativeL2ErrorStopCriterion(BaseStopCriterion):
    """Stops when relative L^2 error is below the set threshold"""

    def __init__(
        self,
        min_iterations: int = 1,
        max_iterations: int = 50,
        threshold: float = 1e-2,
        epsilon: float = 1e-5,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )
        self._threshold = threshold
        self._epsilon = epsilon

    def _should_stop_after(
        self, previous_iteration: Tensor, current_iteration: Tensor, iteration_to_end: int
    ) -> bool:
        error = (current_iteration - previous_iteration).norm() / (
            self._epsilon + current_iteration.norm().item()
        )
        return bool(error < self._threshold)


class AndersonSolverArguments(NamedTuple):
    """Arguments for Anderson solver"""

    memory_length: int = 4
    beta: float = 1.0
    matrix_epsilon: float = 1e-4


class AndersonSolver(IFixedPointSolver):
    """Anderson fixed point solver

    The implementation is based code from the NeurIPS 2020 tutorial by Zico
    Kolter, David Duvenaud, and Matt Johnson.
    (http://implicit-layers-tutorial.org/deep_equilibrium_models/)
    """

    def __init__(
        self,
        stop_criterion: Optional[IFixedPointStopCriterion] = None,
        arguments: Optional[AndersonSolverArguments] = None,
    ) -> None:
        self._stop_criterion = (
            MaxElementWiseAbsStopCriterion() if stop_criterion is None else stop_criterion
        )
        self._arguments = AndersonSolverArguments() if arguments is None else arguments

    def solve(
        self, fixed_point_function: Callable[[Tensor, Tensor], None], initial_value: Tensor
    ) -> Tensor:
        if self._stop_criterion.should_stop_before(0):
            logger.debug("Anderson fixed point iteration returned directly initial value.")
            return initial_value
        initial_value = initial_value.detach()
        batch_size = initial_value.size(0)
        data_shape = initial_value.shape[1:]
        input_memory = zeros(
            (batch_size, self._arguments.memory_length) + data_shape,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        output_memory = zeros_like(input_memory)
        input_memory[:, 0] = initial_value
        fixed_point_function(initial_value, output_memory[:, 0])
        if self._stop_criterion.should_stop_after(
            previous_iteration=input_memory[:, 0],
            current_iteration=output_memory[:, 0],
            iteration_to_end=0,
        ) or self._stop_criterion.should_stop_before(1):
            logger.debug("Anderson fixed point iteration stopped after 1 iteration")
            return output_memory[:, 0].clone()
        input_memory[:, 1] = output_memory[:, 0]
        fixed_point_function(output_memory[:, 0], output_memory[:, 1])
        if self._stop_criterion.should_stop_after(
            previous_iteration=input_memory[:, 1],
            current_iteration=output_memory[:, 1],
            iteration_to_end=1,
        ):
            logger.debug("Anderson fixed point iteration stopped after 2 iterations")
            return output_memory[:, 1].clone()
        coefficients_matrix = zeros(
            batch_size,
            self._arguments.memory_length + 1,
            self._arguments.memory_length + 1,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        coefficients_matrix[:, 0, 1:] = coefficients_matrix[:, 1:, 0] = 1
        solving_target = zeros(
            batch_size,
            self._arguments.memory_length + 1,
            1,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        solving_target[:, 0] = 1
        index = n_iterations = 2
        while not self._stop_criterion.should_stop_before(index):
            current_memory_length = min(index, self._arguments.memory_length)
            step_differences = (
                output_memory[:, :current_memory_length] - input_memory[:, :current_memory_length]
            ).view(batch_size, current_memory_length, -1)
            coefficients_matrix[:, 1 : current_memory_length + 1, 1 : current_memory_length + 1] = (
                bmm(step_differences, step_differences.transpose(1, 2))
                + self._arguments.matrix_epsilon
                * eye(
                    current_memory_length, dtype=initial_value.dtype, device=initial_value.device
                )[None]
            )
            del step_differences
            alpha = solve(
                coefficients_matrix[:, : current_memory_length + 1, : current_memory_length + 1],
                solving_target[:, : current_memory_length + 1],
            )[:, 1 : current_memory_length + 1, 0]
            input_memory[:, index % self._arguments.memory_length] = (
                self._arguments.beta
                * (
                    alpha[:, None]
                    @ output_memory[:, :current_memory_length].view(
                        batch_size, current_memory_length, -1
                    )
                )[:, 0]
            ).view_as(initial_value)
            if self._arguments.beta != 1.0:
                input_memory[:, index % self._arguments.memory_length] += (
                    (1 - self._arguments.beta)
                    * (
                        alpha[:, None]
                        @ input_memory[:, :current_memory_length].view(
                            batch_size, current_memory_length, -1
                        )
                    )[:, 0]
                ).view_as(initial_value)
            del alpha
            fixed_point_function(
                input_memory[:, index % self._arguments.memory_length],
                output_memory[:, index % self._arguments.memory_length]
            )
            n_iterations += 1
            if self._stop_criterion.should_stop_after(
                previous_iteration=input_memory[:, index % self._arguments.memory_length],
                current_iteration=output_memory[:, index % self._arguments.memory_length],
                iteration_to_end=index,
            ):
                break
            index = index + 1
        logger.debug("Anderson fixed point iteration stopped after %d iterations", n_iterations)
        return output_memory[:, index % self._arguments.memory_length].clone()


class NaiveSolverArguments(NamedTuple):
    """Arguments for naive solver"""

    max_error: float = 1e-2
    max_iterations: int = 100
    min_iterations: int = 15


class NaiveSolver(IFixedPointSolver):
    """Naive fixed point solver"""

    def __init__(
        self,
        stop_criterion: Optional[IFixedPointStopCriterion] = None,
    ) -> None:
        self._stop_criterion = (
            MaxElementWiseAbsStopCriterion() if stop_criterion is None else stop_criterion
        )

    def solve(
        self, fixed_point_function: Callable[[Tensor, Tensor], None], initial_value: Tensor
    ) -> Tensor:
        cache = empty(
            (2,) + initial_value.shape, dtype=initial_value.dtype, device=initial_value.device
        )
        cache[0] = initial_value
        n_iterations = 0
        while not self._stop_criterion.should_stop_before(n_iterations):
            fixed_point_function(cache[n_iterations % 2], cache[(n_iterations + 1) % 2])
            n_iterations += 1
            if self._stop_criterion.should_stop_after(
                previous_iteration=cache[n_iterations % 2],
                current_iteration=cache[(n_iterations + 1) % 2],
                iteration_to_end=n_iterations - 1,
            ):
                break
        logger.debug("Naive fixed point iteration stopped after %d iterations", n_iterations)
        return cache[n_iterations % 2].clone()


class EmptySolver(IFixedPointSolver):
    """Empty fixed point solver which returns the initial guess"""

    def solve(
        self, fixed_point_function: Callable[[Tensor, Tensor], None], initial_value: Tensor
    ) -> Tensor:
        return initial_value
