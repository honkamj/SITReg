"""Spatial derivation related implementations"""

from itertools import product
from typing import Callable

from torch import Tensor
from torch import arange, clamp, eye, floor, index_select, prod, stack
from torch import sum as torch_sum
from torch import tensor

from util.optional import optional_add


def spatial_derivatives_on_grid(
    volume: Tensor,
    points: Tensor,
) -> Tensor:
    """Calculate spatial derivatives at given points assuming linear
    interpolation between given volume values

    Args:
        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        points: Interpolation locations, Tensor with shape (batch_size, n_dims, *target_shape)

    Returns:
        Tensor with shape (batch_size, n_channels, n_dims, *target_shape)
    """
    device = volume.device
    batch_size = volume.size(0)
    n_dims = points.size(1)
    n_channels = volume.size(1)
    volume_shape = volume.shape[2:]
    target_shape = points.shape[2:]
    n_points = int(prod(tensor(target_shape)))

    scale = tensor(volume_shape, dtype=volume.dtype, device=device).view(1, n_dims, 1) - 1
    points_flattened = points.view(batch_size, n_dims, n_points)
    points_scaled = points_flattened / scale
    points_clamped = clamp(points_scaled, min=0, max=1) * scale

    points_lower = floor(points_clamped)
    points_upper = floor(clamp((points_flattened + 1) / scale, min=0, max=1) * scale)

    corner_indices = [points_lower.int(), points_upper.int()]

    local_coordinates = points_clamped - points_lower

    dim_product_list = [1]
    for dim in range(0, n_dims):
        dim_product_list.append(dim_product_list[dim] * volume_shape[n_dims - dim - 1])
    dim_product = tensor(list(reversed(dim_product_list)), device=device)
    dim_inclusion_matrix = eye(n_dims, device=device).view(1, n_dims, n_dims, 1)
    dim_exclusion_matrix = 1 - dim_inclusion_matrix

    jacobian_matrices: Tensor | None = None
    for corner_points in product((0, 1), repeat=n_dims):
        volume_indices = stack(
            [corner_indices[corner_points[dim]][:, dim] for dim in range(n_dims)], dim=-1
        )
        volume_indices_1d = torch_sum(volume_indices * dim_product[1:], dim=-1)
        batch_volume_indices_1d = (
            volume_indices_1d.T + arange(batch_size, device=device) * dim_product[0]
        ).T.reshape(n_points * batch_size)
        volume_1d = volume.transpose(0, 1).reshape(n_channels, -1)
        volume_values = (
            index_select(volume_1d, dim=1, index=batch_volume_indices_1d)
            .view(n_channels, batch_size, n_points)
            .transpose(0, 1)
        )
        corner_points_tensor = tensor(corner_points, device=volume.device).view(1, n_dims, 1)
        weights = (1 - corner_points_tensor) * (
            1 - local_coordinates
        ) + corner_points_tensor * local_coordinates
        dim_multiplier = 2 * corner_points_tensor.view(1, 1, n_dims, 1) - 1
        weight_products = (
            prod(
                dim_inclusion_matrix
                + weights.view(batch_size, n_dims, 1, n_points) * dim_exclusion_matrix,
                dim=1,
                keepdim=True,
            )
            * dim_multiplier
        )
        jacobian_matrices = optional_add(
            jacobian_matrices,
            weight_products * volume_values.view(batch_size, n_channels, 1, n_points),
        )
    assert jacobian_matrices is not None
    jacobian_matrices = jacobian_matrices.view(batch_size, n_channels, n_dims, *target_shape)
    return jacobian_matrices


def estimate_spatial_derivatives(
    mapping: Callable[[Tensor], Tensor], points: Tensor, perturbation: float = 1e-7
) -> Tensor:
    """Estimate spatial derivatives at given points using small perturbations
    along each dimension

    Args:
        mapping: Mapping for which to estimate derivatives
        points: Interpolation locations, Tensor with shape (batch_size, n_dims, *target_shape)
        perturbation: Size of perturbation used for estimation

    Returns:
        Tensor with shape (batch_size, *channel_dims, n_dims, *target_shape)
    """
    n_dims = points.size(1)
    perturbations = eye(n_dims, device=points.device, dtype=points.dtype) * perturbation
    target_shape = points.shape[2:]
    points_perturbed = (
        points[:, :, None, ...]
        + perturbations[
            (
                None,
                ...,
            )
            + (None,) * len(target_shape)
        ]
    )
    perturbed_values = mapping(points_perturbed)
    non_perturbed_values = mapping(points)
    gradients = (
        perturbed_values
        - non_perturbed_values[(...,) + (None,) + len(target_shape) * (slice(None),)]
    ) / perturbation
    return gradients
