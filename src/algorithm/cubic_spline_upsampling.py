"""Cubic spline upsampling algorithm with repeating border condition

The implementation is based on nitorch by Yael Balbastre and Mikael Brudfors.
The nitorch functions themselves are ported from implementations by John Ashburner
which are further ports from Philippe Thevenaz's code.

Codes from scipy
(https://github.com/scipy/scipy/blob/main/scipy/ndimage/src/ni_splines.c) and
TorchIR (https://github.com/BDdeVos/TorchIR) have been also used as an inspiration

Link to nitorch: https://github.com/balbasty/nitorch
"""
from math import sqrt
from typing import List, Optional, Sequence, Union

from torch import Tensor
from torch import abs as torch_abs
from torch import arange
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import matmul, tensor
from torch.autograd import Function
from torch.jit import script
from torch.nn import Module, Parameter, ParameterList
from torch.nn.functional import conv_transpose1d, pad

from util.dimension_order import move_axis


class CubicSplineUpsampling(Module):
    """Cubic spline upsampling

    New sampled locations will correspond to the align_corners = False
    setting in torch.nn.functional.interpolate. That is, each pixel/voxel
    value is assumed to be at the center of the corresponding square/cube.

    Arguments:
        upsampling_factor: Dimension size is multiplied by this
    """

    def __init__(
        self, upsampling_factor: Union[int, Sequence[int]], dtype: torch_dtype | None = None
    ) -> None:
        super().__init__()
        if isinstance(upsampling_factor, Sequence):
            self.upsampling_kernels = ParameterList(
                [
                    Parameter(cubic_bspline_kernel_1d(factor, dtype=dtype), requires_grad=False)
                    for factor in upsampling_factor
                ]
            )
        else:
            self.upsampling_kernels = ParameterList(
                [
                    Parameter(
                        cubic_bspline_kernel_1d(upsampling_factor, dtype=dtype), requires_grad=False
                    )
                ]
            )
        self._upsampling_factor = upsampling_factor

    def forward(
        self,
        volume: Tensor,
        apply_prefiltering: bool = True,
        prefilter_inplace: bool = False,
    ) -> Tensor:
        """Cubic spline upsample a volume

        Args:
            volume: Tensor with shape ([batch_size, n_channels, ]dim_1, ..., dim_{n_dims})
            apply_prefiltering: Whether to apply prefiltering
            prefilter_inplace: Whether to perform the prefiltering step in-place
        """
        first_spatial_dim = min(volume.ndim - 1, 2)
        n_dims = volume.ndim - first_spatial_dim
        if isinstance(self._upsampling_factor, int):
            upsampling_factors: Sequence[int] = n_dims * [self._upsampling_factor]
            upsampling_kernels: list[Parameter] = n_dims * list(self.upsampling_kernels)
        else:
            upsampling_factors = self._upsampling_factor
            upsampling_kernels = list(self.upsampling_kernels)
        if apply_prefiltering:
            if upsampling_factors == n_dims * [1]:
                return volume
            upsampled = cubic_spline_coefficients(volume, inplace=prefilter_inplace)
        else:
            upsampled = volume
        for dim, upsampling_factor, upsampling_kernel in zip(
            range(first_spatial_dim, upsampled.ndim), upsampling_factors, upsampling_kernels
        ):
            upsampled = _transposed_conv1d(
                upsampled,
                dim=dim,
                kernel=upsampling_kernel,
                stride=upsampling_factor,
                implicit_padding=7 * upsampling_factor // 2,
                padding=2,
            )
        return upsampled


def cubic_spline_coefficients(volume: Tensor, inplace: bool = False) -> Tensor:
    """Calculate spline coefficients for cubic spline interpolation"""
    return _CubicSplineCoefficients.apply(volume, inplace)


class _CubicSplineCoefficients(Function):  # pylint: disable=abstract-method
    """Cubic spline coefficient function with custom backward implementation

    The function has identical forward and backward passes.
    """

    @staticmethod
    def forward(ctx, volume: Tensor, inplace: bool = False) -> Tensor:  # type: ignore # pylint: disable=arguments-differ
        return _cubic_spline_coefficients(volume, inplace)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore # pylint: disable=arguments-differ
        grad_needed, _ = ctx.needs_input_grad
        if grad_needed:
            return _cubic_spline_coefficients(grad_output, inplace=False), None
        return None, None


@script
def _transposed_conv1d(
    input_tensor: Tensor, dim: int, kernel: Tensor, stride: int, implicit_padding: int, padding: int
) -> Tensor:
    """Performs same 1d convolution over every channel over one dimension

    Args:
        input_tensor: Tensor with shape (*any_shape)
        kernel: Tensor with shape (kernel_size,)
        dim: Dimension over which the convolution is performed
        stride: Stride of the transposed convolution
        implicit_padding: Implicit padding of the transposed convolution
        padding: Explicit padding added to both sides of the input
    """
    dim_size = input_tensor.size(dim)
    input_tensor = move_axis(input_tensor, dim, -1)
    dim_excluded_shape = input_tensor.shape[:-1]
    input_tensor = input_tensor.reshape(-1, 1, dim_size)
    input_tensor = pad(input_tensor, pad=(padding, padding), mode="replicate")
    convolved = conv_transpose1d(
        input_tensor, kernel[None, None], bias=None, stride=stride, padding=implicit_padding
    ).reshape(dim_excluded_shape + (-1,))
    return move_axis(convolved, -1, dim)


@script
def cubic_spline_1d(points: Tensor) -> Tensor:
    """Cubic spline basis function"""
    points_abs = torch_abs(points)
    output = (2 / 3 + (0.5 * points_abs - 1) * points_abs**2) * (points_abs < 1)
    output = -((points_abs - 2) ** 3) / 6 * ((points_abs >= 1) & (points_abs < 2)) + output
    return output


@script
def cubic_bspline_kernel_1d(
    upsampling_factor: int,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> Tensor:
    """Cubic B-spline kernel for specified upsampling factor"""
    is_odd = upsampling_factor % 2
    kernel_size = 4 * upsampling_factor - is_odd
    step_size = 1 / upsampling_factor
    start = (1 + is_odd) / (2 * upsampling_factor) - 2
    points = arange(kernel_size, dtype=dtype, device=device) * step_size + start
    return cubic_spline_1d(points)


@script
def _get_gain(poles: List[float]) -> float:
    gain: float = 1.0
    for pole in poles:
        gain *= (1.0 - pole) * (1.0 - 1.0 / pole)
    return gain


@script
def _bound_causal(coefficients: Tensor, pole: float, dim: int = 0) -> Tensor:
    dim_size = coefficients.size(0)
    pole_tensor = tensor(pole, dtype=coefficients.dtype, device=coefficients.device)
    pole_pows = pole_tensor.pow(
        arange(0, dim_size, dtype=coefficients.dtype, device=coefficients.device)
    ) + pole_tensor.pow(
        arange(
            2 * dim_size - 1, dim_size - 1, -1, dtype=coefficients.dtype, device=coefficients.device
        )
    )
    output = matmul(
        move_axis(coefficients, dim, -1).unsqueeze(-2), pole_pows.unsqueeze(-1)
    ).squeeze(-1)
    output = output * pole / (1 - pow(pole, 2 * dim_size)) + coefficients[0].unsqueeze(-1)
    return output.squeeze(-1)


@script
def _bound_anticausal(coefficients: Tensor, pole: float, dim: int = 0) -> Tensor:
    output = move_axis(coefficients, dim, 0)[-1] * (pole / (pole - 1))
    return output


@script
def _cubic_spline_coefficients_1d(coefficients: Tensor, dim: int = -1) -> Tensor:
    """Calculates spline coefficients for one dimension in-place"""
    if coefficients.shape[dim] == 1:
        return coefficients
    cubic_pole = sqrt(3.0) - 2.0
    gain = _get_gain(cubic_pole)
    coefficients *= gain
    coefficients = move_axis(coefficients, dim, 0)
    dim_size = coefficients.size(0)
    coefficients[0] = _bound_causal(coefficients, cubic_pole, dim=0)
    for index in range(1, dim_size):
        coefficients[index].add_(coefficients[index - 1], alpha=cubic_pole)
    coefficients[-1] = _bound_anticausal(coefficients, cubic_pole, dim=0)
    for index in range(dim_size - 2, -1, -1):
        coefficients[index].neg_().add_(coefficients[index + 1]).mul_(cubic_pole)
    output = move_axis(coefficients, 0, dim)
    return output


@script
def _cubic_spline_coefficients(volume: Tensor, inplace: bool = False) -> Tensor:
    """Compute the interpolating spline coefficients for cubic spline

    Args:
        volume: Volume to interpolate, Tensor with shape
            ([batch_size, n_channels, ]dim_1, ..., dim_{n_dims})
        inplace: Whether to perform the operation inplace

    Returns: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
    """
    if not inplace:
        volume = volume.clone()
    first_spatial_dim = min(volume.ndim - 1, 2)
    for dim in range(first_spatial_dim, volume.ndim):
        volume = _cubic_spline_coefficients_1d(volume, dim=dim)
    return volume
