"""Spectral norm estimation using power iteration"""


from typing import Sequence
from deformation_inversion_layer.fixed_point_iteration import (
    BaseCountingStopCriterion,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)
from deformation_inversion_layer.interface import FixedPointSolver
from torch import Tensor, cat, matmul, randn, sign
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable
from torch.nn.functional import normalize

from util.dimension_order import (
    broadcast_tensors_by_leading_dims,
    channels_last,
    index_by_channel_dims,
    merged_batch_dimensions,
)


class SpectralNormStopCriterionMixin(BaseCountingStopCriterion):
    """Stop criterion which directly checks the convergence of the spectral norm"""

    def __init__(self, n_rows: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_rows = n_rows
        self._gradient_cache: dict[int, Tensor] = {}

    def _gradient(self, singular_vectors: Tensor) -> Tensor:
        left_singular_vector = singular_vectors[:, : self._n_rows]
        right_singular_vector = singular_vectors[:, self._n_rows :]
        return matmul(left_singular_vector, right_singular_vector.transpose(1, 2))

    def _should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        if len(previous_iterations) == 0:
            return False
        previous_iteration = previous_iterations[0]
        if n_earlier_iterations - 1 in self._gradient_cache:
            previous_iteration_gradient = self._gradient_cache[n_earlier_iterations - 1]
        else:
            previous_iteration_gradient = self._gradient(previous_iteration)
        current_iteration_gradient = self._gradient(current_iteration)
        if self._should_check_convergence(n_earlier_iterations + 1):
            self._gradient_cache[n_earlier_iterations] = current_iteration_gradient
        return super()._should_stop(
            current_iteration_gradient,
            [previous_iteration_gradient],
            n_earlier_iterations,
        )

    def should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        should_stop = super().should_stop(
            current_iteration=current_iteration,
            previous_iterations=previous_iterations,
            n_earlier_iterations=n_earlier_iterations,
        )
        if should_stop:
            self._gradient_cache.clear()
        elif n_earlier_iterations - 1 in self._gradient_cache:
            del self._gradient_cache[n_earlier_iterations - 1]
        return should_stop


class SpectralNormRelativeL2ErrorStopCriterion(
    SpectralNormStopCriterionMixin, RelativeL2ErrorStopCriterion
):
    """Spectral norm version of relative L^2 error stop criterion"""

    def __init__(
        self,
        n_rows: int,
        min_iterations: int = 1,
        max_iterations: int = 50,
        threshold: float = 1e-2,
        epsilon: float = 1e-5,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            n_rows=n_rows,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            epsilon=epsilon,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )


class SpectralNormMaxElementWiseAbsStopCriterion(
    SpectralNormStopCriterionMixin, MaxElementWiseAbsStopCriterion
):
    """Spectral norm version of relative L^2 error stop criterion"""

    def __init__(
        self,
        n_rows: int,
        min_iterations: int = 1,
        max_iterations: int = 50,
        threshold: float = 1e-2,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            n_rows=n_rows,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )


def calculate_spectral_norm_with_power_iteration(
    matrix: Tensor,
    solver: FixedPointSolver,
    initial_left_singular_vector: Tensor | None = None,
    initial_right_singular_vector: Tensor | None = None,
) -> Tensor:
    """Compute spectral norm of a matrix with power iteration

    Args:
        matrix: Tensor with shape (batch_size, n_rows, n_cols, *any_shape)
        solver: Fixed point solver to use
        initial_left_singular_vector: Initial guess for left singular vector,
            Tensor with shape ([batch_size, ]n_rows, *any_shape)
        initial_right_singular_vector: Initial guess for right singular vector,
            Tensor with shape ([batch_size, ]n_cols, *any_shape)

    Returns:
        Tensor with shape (batch_size, 1, *any_shape)
    """
    return _SpectralNormPowerIteration.apply(
        matrix, solver, initial_left_singular_vector, initial_right_singular_vector
    )


class _SpectralNormPowerIteration(Function):  # pylint: disable=abstract-method
    """Compute spectral norm with power iteration"""

    @channels_last(
        {
            "matrix": 2,
            "initial_left_singular_vector": 1,
            "initial_right_singular_vector": 1,
        },
        1,
    )
    @merged_batch_dimensions(
        {
            "matrix": 2,
            "initial_left_singular_vector": 1,
            "initial_right_singular_vector": 1,
        },
        1,
    )
    @staticmethod
    def _calculate_spectral_norm(
        matrix: Tensor,
        initial_left_singular_vector: Tensor,
        initial_right_singular_vector: Tensor,
        solver: FixedPointSolver,
    ) -> tuple[Tensor, Tensor, Tensor]:
        n_rows = initial_left_singular_vector.size(1)
        combined_singular_vectors = cat(
            (initial_left_singular_vector, initial_right_singular_vector), dim=1
        )[..., None]
        del initial_left_singular_vector
        del initial_right_singular_vector

        def _spectral_norm_power_iteration_step(
            singular_vectors: Tensor, out: Tensor
        ) -> None:
            normalize(
                matmul(
                    matrix.transpose(1, 2),
                    singular_vectors[:, :n_rows],
                    out=combined_singular_vectors[:, n_rows:],
                ),
                dim=1,
                out=out[:, n_rows:],
            )
            normalize(
                matmul(
                    matrix,
                    singular_vectors[:, n_rows:],
                    out=combined_singular_vectors[:, :n_rows],
                ),
                dim=1,
                out=out[:, :n_rows],
            )

        singular_vectors = solver.solve(
            fixed_point_function=_spectral_norm_power_iteration_step,
            initial_value=combined_singular_vectors,
        )
        left_singular_vector = singular_vectors[:, :n_rows]
        right_singular_vector = singular_vectors[:, n_rows:]
        spectral_norm = matmul(
            left_singular_vector.transpose(1, 2), matmul(matrix, right_singular_vector)
        )[..., 0]
        return (
            spectral_norm.abs(),
            left_singular_vector[..., 0] * sign(spectral_norm),
            right_singular_vector[..., 0],
        )

    @staticmethod
    def forward(  # type: ignore # pylint: disable=arguments-differ
        ctx: FunctionCtx,
        matrix: Tensor,
        solver: FixedPointSolver,
        initial_left_singular_vector: Tensor | None,
        initial_right_singular_vector: Tensor | None,
    ):
        first_channel_index = index_by_channel_dims(
            n_total_dims=matrix.ndim, channel_dim_index=0, n_channel_dims=2
        )
        second_channel_index = index_by_channel_dims(
            n_total_dims=matrix.ndim, channel_dim_index=1, n_channel_dims=2
        )
        initial_left_singular_vector = (
            randn(
                matrix.shape[:second_channel_index]
                + matrix.shape[second_channel_index + 1 :],
                dtype=matrix.dtype,
                device=matrix.device,
            )
            if initial_left_singular_vector is None
            else initial_left_singular_vector
        )
        initial_right_singular_vector = (
            randn(
                matrix.shape[:first_channel_index]
                + matrix.shape[first_channel_index + 1 :],
                dtype=matrix.dtype,
                device=matrix.device,
            )
            if initial_right_singular_vector is None
            else initial_right_singular_vector
        )
        (
            matrix,
            initial_left_singular_vector,
            initial_right_singular_vector,
        ) = broadcast_tensors_by_leading_dims(
            (matrix, initial_left_singular_vector, initial_right_singular_vector),
            num_leading_dims=(2, 1, 1),
        )
        (
            spectral_norm,
            left_singular_vector,
            right_singular_vector,
        ) = _SpectralNormPowerIteration._calculate_spectral_norm(
            matrix=matrix,
            initial_left_singular_vector=initial_left_singular_vector,
            initial_right_singular_vector=initial_right_singular_vector,
            solver=solver,
        )
        grad_needed, _, _, _ = ctx.needs_input_grad  # type: ignore
        if grad_needed:
            ctx.save_for_backward(left_singular_vector, right_singular_vector)
        return spectral_norm

    @channels_last(1, 2)
    @merged_batch_dimensions(1, 2)
    @staticmethod
    def _calculate_spectral_norm_backward(
        grad_output: Tensor,
        left_singular_vector: Tensor,
        right_singular_vector: Tensor,
    ) -> Tensor:
        return grad_output[:, None] * matmul(
            left_singular_vector[..., None], right_singular_vector[:, None]
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        grad_needed, _, _, _ = ctx.needs_input_grad
        if grad_needed:
            left_singular_vector: Tensor
            right_singular_vector: Tensor
            (
                left_singular_vector,
                right_singular_vector,
            ) = ctx.saved_tensors
            gradient = _SpectralNormPowerIteration._calculate_spectral_norm_backward(
                grad_output=grad_output,
                left_singular_vector=left_singular_vector,
                right_singular_vector=right_singular_vector,
            )
            return gradient, None, None, None
        return None, None, None, None
