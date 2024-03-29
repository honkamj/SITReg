"""Algorithm for computing upper bound which when fulfilled ensure invertibility
of generated deformations"""

from itertools import product
from math import ceil
from typing import Sequence
from numpy import prod

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import stack, tensor
from torch.nn.functional import pad
from tqdm import tqdm  # type: ignore

from algorithm.cubic_spline_upsampling import cubic_bspline_kernel_1d


def compute_max_control_point_value(
    upsampling_factors: Sequence[int],
    dtype: torch_dtype | None = None,
    device: torch_device | None = None,
    show_progress_bar: bool = False,
) -> Tensor:
    """Compute cublic B-spline control point max value ensuring invertibility
    for different cubic spline upsampling rates and dimensionalities

    Having control point absolute values to be lower than the resulting bound
    ensures invertibility by the deformation inversion layer (and is also a
    tight bound for invertibility in general) for the deformation generated by
    the control points when the final deformation is sampled with the frequency
    of the upsampling factor (compared to the control point grid density).

    Args:
        upsampling_factor: Upsampling factor with respect to control point density
        n_dims: Dimensionality of the deformation
        dtype: Pytorch dtype to use in the computations
        device: Pytorch device to use in the computations
        progress_bar: Show progress bar

    Returns: 0-dimensional Tensor with the max value
    """
    n_dims = len(upsampling_factors)
    kernels = [cubic_bspline_kernel_1d(factor, dtype, device) for factor in upsampling_factors]
    padded_kernels = [pad(kernel, pad=(1, 1)) for kernel in kernels]
    kernel_derivatives = [kernel[1:] - kernel[:-1] for kernel in padded_kernels]
    upper_limit = tensor(0.0, dtype=dtype, device=device)
    locations = product(*(range(int(ceil((factor + 1) / 2))) for factor in upsampling_factors))
    if show_progress_bar:
        n_locations = prod([int(ceil((factor + 1) / 2)) for factor in upsampling_factors])
        progress_bar = tqdm(range(n_locations), unit="locations")
    for start_indices in locations:
        if show_progress_bar:
            progress_bar.update(1)
        derivatives = []
        for derivative_dim in range(n_dims):
            sliced_kernels = [
                kernel_derivatives[dim][start_indices[dim] :: upsampling_factors[dim]]
                if dim == derivative_dim
                else padded_kernels[dim][:-1][start_indices[dim] :: upsampling_factors[dim]]
                for dim in range(n_dims)
            ]
            nd_kernel_derivative = sliced_kernels[0]
            for sliced_kernel in sliced_kernels[1:]:
                nd_kernel_derivative = nd_kernel_derivative[..., None] * sliced_kernel
            derivatives.append(nd_kernel_derivative * upsampling_factors[derivative_dim])
        upper_limit_candidate = stack(derivatives, dim=0).sum(dim=0).abs().sum()
        if upper_limit_candidate > upper_limit:
            upper_limit = upper_limit_candidate
    return 1 / upper_limit
