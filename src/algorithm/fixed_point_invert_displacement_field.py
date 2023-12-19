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
    coordinates: Optional[Tensor] = None,
) -> Tensor:
    """Fixed point invert displacement field

    Args:
        displacement_field: Displacement field to invert
        arguments: Arguments for fixed point inversion
        initial_guess: Initial guess for inverted displacement field
        coordinates: Voxel coordinates at which to invert (usually given as a grid)
    """
    return _FixedPointInvertDisplacementField.apply(
        displacement_field, arguments, initial_guess, coordinates
    )


class _FixedPointInvertDisplacementField(Function):  # pylint: disable=abstract-method
    """Fixed point invert displacement field"""

    @staticmethod
    def _forward_fixed_point_iteration_step(
        inverted_displacement_field: Tensor,
        displacement_field: Tensor,
        interpolator: IInterpolator,
        coordinates: Tensor,
    ) -> Tensor:
        return -interpolator(
            volume=displacement_field,
            coordinates=coordinates + inverted_displacement_field,
        )

    @staticmethod
    def _forward_fixed_point_mapping(
        inverted_displacement_field: Tensor,
        out: Tensor,
        displacement_field: Tensor,
        interpolator: IInterpolator,
        coordinates: Tensor,
    ) -> None:
        out[:] = _FixedPointInvertDisplacementField._forward_fixed_point_iteration_step(
            inverted_displacement_field=inverted_displacement_field,
            displacement_field=displacement_field,
            interpolator=interpolator,
            coordinates=coordinates,
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
        coordinates: Optional[Tensor],
    ):
        dtype = (
            displacement_field.dtype
            if arguments.forward_dtype is None
            else arguments.forward_dtype
        )
        type_converted_displacement_field = displacement_field.to(dtype=dtype)
        if coordinates is None:
            type_converted_coordinates = generate_voxel_coordinate_grid(
                displacement_field.shape[2:], displacement_field.device, dtype=dtype
            )
        elif displacement_field.dtype != coordinates.dtype:
            raise ValueError(
                f'DType {coordinates.dtype} of input "coordinates" does not match '
                f'DType {displacement_field.dtype} of input "displacement_field"'
            )
        else:
            type_converted_coordinates = coordinates.to(dtype=dtype)
        if initial_guess is None and coordinates is None:
            type_converted_initial_guess = -type_converted_displacement_field
        elif initial_guess is None:
            type_converted_initial_guess = zeros_like(type_converted_coordinates)
        elif displacement_field.dtype != initial_guess.dtype:
            raise ValueError(
                f'DType {initial_guess.dtype} of input "initial_guess" does not match '
                f'DType {displacement_field.dtype} of input "displacement_field"'
            )
        else:
            type_converted_initial_guess = initial_guess.to(dtype=dtype)
        inverted_displacement_field = arguments.forward_solver.solve(
            partial(
                _FixedPointInvertDisplacementField._forward_fixed_point_mapping,
                displacement_field=type_converted_displacement_field,
                coordinates=type_converted_coordinates,
                interpolator=arguments.interpolator,
            ),
            initial_value=type_converted_initial_guess,
        ).to(displacement_field.dtype)
        (
            displacement_field_grad_needed,
            _,
            _,
            coordinates_grad_needed,
        ) = ctx.needs_input_grad  # type: ignore
        if displacement_field_grad_needed or coordinates_grad_needed:
            tensors_to_save = [displacement_field, inverted_displacement_field]
            if coordinates is not None:
                tensors_to_save.append(coordinates)
            ctx.save_for_backward(*tensors_to_save)
            ctx.arguments = arguments  # type: ignore
            ctx.dtype = dtype  # type: ignore
            ctx.has_coordinates = coordinates is not None  # type: ignore
        return inverted_displacement_field

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        (
            displacement_field_grad_needed,
            _,
            _,
            coordinates_grad_needed,
        ) = ctx.needs_input_grad
        if displacement_field_grad_needed or coordinates_grad_needed:
            displacement_field: Tensor = ctx.saved_tensors[0]
            inverted_displacement_field: Tensor = ctx.saved_tensors[1]
            coordinates: Tensor | None = (
                ctx.saved_tensors[2] if ctx.has_coordinates else None
            )
            arguments: DisplacementFieldInversionArguments = ctx.arguments
            del ctx
            dtype = (
                displacement_field.dtype
                if arguments.backward_dtype is None
                else arguments.backward_dtype
            )
            original_dtype = displacement_field.dtype
            displacement_field = displacement_field.to(dtype).detach()
            if coordinates is None:
                coordinates = generate_voxel_coordinate_grid(
                    displacement_field.shape[2:], displacement_field.device, dtype=dtype
                )
            else:
                coordinates = coordinates.to(dtype=dtype)
            inverted_displacement_field = inverted_displacement_field.to(dtype).detach()
            grad_output = grad_output.to(dtype)
            if arguments.backward_solver is None:
                raise RuntimeError("Backward solver not specified!")
            with enable_grad():
                displacement_field.requires_grad_(displacement_field_grad_needed)
                coordinates.requires_grad_(coordinates_grad_needed)
                inverted_displacement_field.requires_grad_(True)
                forward_fixed_point_output = _FixedPointInvertDisplacementField._forward_fixed_point_iteration_step(
                    inverted_displacement_field=inverted_displacement_field,
                    displacement_field=displacement_field,
                    interpolator=arguments.interpolator,
                    coordinates=coordinates,
                )
                fixed_point_solved_gradient = arguments.backward_solver.solve(
                    partial(
                        _FixedPointInvertDisplacementField._backward_fixed_point_mapping,
                        inverted_displacement_field=inverted_displacement_field,
                        forward_fixed_point_output=forward_fixed_point_output,
                        grad_output=grad_output,
                    ),
                    initial_value=zeros_like(inverted_displacement_field),
                )
                displacement_field.requires_grad_(displacement_field_grad_needed)
                coordinates.requires_grad_(coordinates_grad_needed)
                inverted_displacement_field.requires_grad_(False)
                differentiated_inputs = []
                if displacement_field_grad_needed:
                    differentiated_inputs.append(displacement_field)
                if coordinates_grad_needed:
                    differentiated_inputs.append(coordinates)
                output_grad = grad(
                    outputs=forward_fixed_point_output,
                    inputs=differentiated_inputs,
                    grad_outputs=fixed_point_solved_gradient,
                    retain_graph=False,
                )
                displacement_field_grad = (
                    output_grad[0].to(dtype=original_dtype)
                    if displacement_field_grad_needed
                    else None
                )
                coordinates_grad = (
                    output_grad[1 if displacement_field_grad_needed else 0].to(
                        dtype=original_dtype
                    )
                    if coordinates_grad_needed
                    else None
                )
                return displacement_field_grad, None, None, coordinates_grad
        return None, None, None, None
