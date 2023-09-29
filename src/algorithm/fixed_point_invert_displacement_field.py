"""Fixed point invert displacement field"""

from functools import partial
from typing import NamedTuple, Optional

from torch import Tensor
from torch import dtype as torch_dtype
from torch import enable_grad, zeros_like
from torch.autograd import grad
from torch.autograd.function import Function, FunctionCtx, once_differentiable

from .dense_deformation import generate_voxel_coordinate_grid
from .interface import IFixedPointSolver, IInterpolator


class DisplacementFieldInversionArguments(NamedTuple):
    """Arguments for displacement field fixed point inversion

    Args:
        interpolator: Interpolator with which to interpolate the input
            displacement field
        forward_solver: Fixed point solver for the forward pass
        forward_dtype: Data type to use for the solver in the forward pass
        backward_solver: Fixed point solver for the backward pass,
            needed for backward pass
        backward_dtype: Data type to use for the solver in the backward pass
    """

    interpolator: IInterpolator
    forward_solver: IFixedPointSolver
    forward_dtype: Optional[torch_dtype] = None
    backward_solver: Optional[IFixedPointSolver] = None
    backward_dtype: Optional[torch_dtype] = None


def fixed_point_invert_displacement_field(
    displacement_field: Tensor,
    arguments: DisplacementFieldInversionArguments,
    initial_guess: Optional[Tensor] = None,
) -> Tensor:
    """Fixed point invert displacement field

    Args:
        displacement_field: Displacement field to invert
        arguments: Arguments for fixed point inversion
        initial_guess: Initial guess for inverted displacement field
    """
    return _FixedPointInvertDisplacementField.apply(displacement_field, arguments, initial_guess)


class _FixedPointInvertDisplacementField(Function):  # pylint: disable=abstract-method
    """Fixed point invert displacement field"""

    @staticmethod
    def _forward_fixed_point_iteration_step(
        inverted_displacement_field: Tensor,
        displacement_field: Tensor,
        interpolator: IInterpolator,
        voxel_coordinate_grid: Tensor,
    ) -> Tensor:
        return -interpolator(
            volume=displacement_field,
            coordinates=voxel_coordinate_grid + inverted_displacement_field,
        )

    @staticmethod
    def _forward_fixed_point_mapping(
        inverted_displacement_field: Tensor,
        out: Tensor,
        displacement_field: Tensor,
        interpolator: IInterpolator,
        voxel_coordinate_grid: Tensor,
    ) -> None:
        out[:] = _FixedPointInvertDisplacementField._forward_fixed_point_iteration_step(
            inverted_displacement_field=inverted_displacement_field,
            displacement_field=displacement_field,
            interpolator=interpolator,
            voxel_coordinate_grid=voxel_coordinate_grid,
        )

    @staticmethod
    def _backward_fixed_point_mapping(
        vjp_estimate: Tensor,
        out: Tensor,
        inverted_displacement_field: Tensor,
        forward_fixed_point_output: Tensor,
        grad_output: Tensor,
    ) -> None:
        out[:] = (
            grad(
                outputs=forward_fixed_point_output,
                inputs=inverted_displacement_field,
                grad_outputs=vjp_estimate,
                retain_graph=True,
            )[0]
            + grad_output
        )

    @staticmethod
    def forward(  # type: ignore # pylint: disable=arguments-differ
        ctx: FunctionCtx,
        displacement_field: Tensor,
        arguments: DisplacementFieldInversionArguments,
        initial_guess: Optional[Tensor],
    ):
        dtype = (
            displacement_field.dtype if arguments.forward_dtype is None else arguments.forward_dtype
        )
        type_converted_displacement_field = displacement_field.to(dtype)
        inverted_displacement_field = arguments.forward_solver.solve(
            partial(
                _FixedPointInvertDisplacementField._forward_fixed_point_mapping,
                displacement_field=type_converted_displacement_field,
                voxel_coordinate_grid=generate_voxel_coordinate_grid(
                    displacement_field.shape[2:], displacement_field.device, dtype=dtype
                ),
                interpolator=arguments.interpolator,
            ),
            initial_value=(
                -type_converted_displacement_field if initial_guess is None else initial_guess
            ),
        ).to(displacement_field.dtype)
        grad_needed, _, _ = ctx.needs_input_grad  # type: ignore
        if grad_needed:
            ctx.save_for_backward(displacement_field, inverted_displacement_field)
            ctx.arguments = arguments  # type: ignore
            ctx.dtype = dtype  # type: ignore
        return inverted_displacement_field

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        grad_needed, _, _ = ctx.needs_input_grad
        if grad_needed:
            displacement_field: Tensor
            inverted_displacement_field: Tensor
            (
                displacement_field,
                inverted_displacement_field,
            ) = ctx.saved_tensors
            arguments: DisplacementFieldInversionArguments = ctx.arguments
            del ctx
            dtype = (
                displacement_field.dtype
                if arguments.backward_dtype is None
                else arguments.backward_dtype
            )
            original_dtype = displacement_field.dtype
            displacement_field = displacement_field.to(dtype).detach()
            inverted_displacement_field = inverted_displacement_field.to(dtype).detach()
            grad_output = grad_output.to(dtype)
            if arguments.backward_solver is None:
                raise RuntimeError("Backward solver not specified!")
            with enable_grad():
                displacement_field.requires_grad_(True)
                inverted_displacement_field.requires_grad_(True)
                forward_fixed_point_output = (
                    _FixedPointInvertDisplacementField._forward_fixed_point_iteration_step(
                        inverted_displacement_field=inverted_displacement_field,
                        displacement_field=displacement_field,
                        interpolator=arguments.interpolator,
                        voxel_coordinate_grid=generate_voxel_coordinate_grid(
                            displacement_field.shape[2:], displacement_field.device, dtype=dtype
                        ),
                    )
                )
                displacement_field.requires_grad_(False)
                fixed_point_solved_gradient = arguments.backward_solver.solve(
                    partial(
                        _FixedPointInvertDisplacementField._backward_fixed_point_mapping,
                        inverted_displacement_field=inverted_displacement_field,
                        forward_fixed_point_output=forward_fixed_point_output,
                        grad_output=grad_output,
                    ),
                    initial_value=zeros_like(grad_output),
                )
                displacement_field.requires_grad_(True)
                inverted_displacement_field.requires_grad_(False)
                output_grad = grad(
                    outputs=forward_fixed_point_output,
                    inputs=displacement_field,
                    grad_outputs=fixed_point_solved_gradient,
                    retain_graph=False,
                )[0]
                return output_grad.to(original_dtype), None, None
        return None, None, None
