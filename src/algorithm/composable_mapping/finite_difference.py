"""Calculate spatial derivatives with respect to the volumes"""

from typing import Optional

from torch import Tensor
from torch import device as torch_device, dtype as torch_dtype

from algorithm.composable_mapping.interface import IComposableMapping, VoxelCoordinateSystem
from algorithm.finite_difference import (
    estimate_spatial_derivatives,
    estimate_spatial_jacobian_matrices,
)


def estimate_spatial_derivatives_for_mapping(
    mapping: IComposableMapping,
    coordinate_system: VoxelCoordinateSystem,
    spatial_dim: int,
    other_dims: str | None = None,
    central: bool = False,
    device: Optional[torch_device] = None,
    dtype: Optional[torch_dtype] = None,
) -> Tensor:
    """Calculate spatial derivative over a dimension

    Args:
        mapping: Derivative is calculate over coordinates of this mapping
            with respect to it's output
        coordinate_system: Defines sampling locations based on which the
            derivative is computed
        other_dims: See option other_dims of algorithm.finite_difference
        central: See option central of algorithm.finite_difference

    Returns:
        if central and same_shape: Tensor with shape
            (batch_size, n_channels, dim_1 - 2, ..., dim_{n_dims} - 2)
        elif central and not same_shape: Tensor with shape
            (batch_size, n_channels, dim_1, ..., dim_{dim} - 2, ..., dim_{n_dims})
        elif not central and same_shape: Tensor with shape
            (batch_size, n_channels, dim_1 - 1, ..., dim_{n_dims} - 1)
        elif not central and not same_shape: Tensor with shape
            (batch_size, n_channels, dim_1, ..., dim_{dim} - 1, ..., dim_{n_dims})
    """
    sampled_values = mapping(coordinate_system.grid)
    volume = sampled_values.generate_values(device=device, dtype=dtype)
    n_channel_dims = len(sampled_values.channels_shape)
    return estimate_spatial_derivatives(
        volume=volume,
        spatial_dim=spatial_dim,
        spacing=coordinate_system.grid_spacing[spatial_dim],
        n_channel_dims=n_channel_dims,
        other_dims=other_dims,
        central=central,
    )


def estimate_spatial_jacobian_matrices_for_mapping(
    mapping: IComposableMapping,
    coordinate_system: VoxelCoordinateSystem,
    other_dims: str = "average",
    central: bool = False,
    device: Optional[torch_device] = None,
    dtype: Optional[torch_dtype] = None,
) -> Tensor:
    """Calculate local Jacobian matrices of a mapping

    Args:
        mapping: Derivative is calculate over coordinates of this mapping
            with respect to it's output
        coordinate_system: Defines sampling locations based on which the
            derivative is computed
        other_dims: See option other_dims of algorithm.finite_difference
        central: See option central of algorithm.finite_difference

    Returns:
        Tensor with shape (batch_size, n_dims, n_dims, dim_1 - 2, ..., dim_{n_dims} - 2)
    """
    sampled_values = mapping(coordinate_system.grid)
    volume = sampled_values.generate_values(device=device, dtype=dtype)
    n_channel_dims = len(sampled_values.channels_shape)
    return estimate_spatial_jacobian_matrices(
        volume=volume,
        spacing=coordinate_system.grid_spacing,
        n_channel_dims=n_channel_dims,
        other_dims=other_dims,
        central=central,
    )
