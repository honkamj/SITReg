"""Functions for handling dense deformations"""

from typing import List, Optional, Tuple

from torch import Tensor
from torch import all as torch_all
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, meshgrid, stack, tensor
from torch.jit import script
from torch.nn.functional import grid_sample

from util.dimension_order import (
    index_by_channel_dims,
    move_channels_first,
    move_channels_last,
    num_spatial_dims,
)
from util.import_util import import_object


@script
def _convert_between_coordinates(
    coordinates: Tensor, volume_shape: Optional[List[int]], to_voxel_coordinates: bool
) -> Tensor:
    channel_dim = index_by_channel_dims(coordinates.ndim, channel_dim_index=0, n_channel_dims=1)
    n_spatial_dims = num_spatial_dims(n_total_dims=coordinates.ndim, n_channel_dims=1)
    n_dims = coordinates.size(channel_dim)
    inferred_volume_shape = coordinates.shape[-n_dims:] if volume_shape is None else volume_shape
    add_spatial_dims_view = (-1,) + n_spatial_dims * (1,)
    volume_shape_tensor = tensor(
        inferred_volume_shape, dtype=coordinates.dtype, device=coordinates.device
    ).view(add_spatial_dims_view)
    coordinate_grid_start = tensor(-1.0, dtype=coordinates.dtype, device=coordinates.device).view(
        add_spatial_dims_view
    )
    coordinate_grid_end = tensor(1.0, dtype=coordinates.dtype, device=coordinates.device).view(
        add_spatial_dims_view
    )
    if to_voxel_coordinates:
        output = (
            (coordinates - coordinate_grid_start)
            / (coordinate_grid_end - coordinate_grid_start)
            * (volume_shape_tensor - 1)
        )
    else:
        output = (
            coordinates / (volume_shape_tensor - 1) * (coordinate_grid_end - coordinate_grid_start)
            + coordinate_grid_start
        )
    return output


@script
def convert_normalized_to_voxel_coordinates(
    coordinates: Tensor, volume_shape: Optional[List[int]] = None
) -> Tensor:
    """Transforms coordinates to voxel coordinates from
    normalized coordinates

    Args:
        coordinates: Tensor with shape (batch_size, n_dims, *grid_shape)
        volume_shape: Shape of the volume, if not given it is assumed to
            equal the grid_shape
        channels_first: Whether to have channels first

    Returns: Tensor with shape (batch_size, n_dims, *grid_shape)
    """
    return _convert_between_coordinates(coordinates, volume_shape, to_voxel_coordinates=True)


@script
def convert_voxel_to_normalized_coordinates(
    coordinates: Tensor, volume_shape: Optional[List[int]] = None
) -> Tensor:
    """Transforms coordinates to normalized coordinates from
    voxel coordinates

    Args:
        coordinates: Tensor with shape (batch_size, n_dims, *grid_shape)
        volume_shape: Shape of the volume, if not given it is assumed to
            equal the grid_shape

    Returns: Tensor with shape (batch_size, n_dims, *grid_shape)
    """
    return _convert_between_coordinates(coordinates, volume_shape, to_voxel_coordinates=False)


@script
def generate_normalized_coordinate_grid(
    shape: List[int], device: torch_device, dtype: Optional[torch_dtype] = None
) -> Tensor:
    """Generate normalized coordinate grid

    Args:
        shape: Shape of the grid
        device: Device of the grid

    Returns: Tensor with shape (1, len(shape), dim_1, ..., dim_{len(shape)})
    """
    axes = [
        linspace(start=-1, end=1, steps=int(dim_size), device=device, dtype=dtype)
        for dim_size in shape
    ]
    coordinates = stack(meshgrid(axes, indexing="ij"), dim=0)
    return coordinates[None]


@script
def generate_voxel_coordinate_grid(
    shape: List[int], device: torch_device, dtype: Optional[torch_dtype] = None
) -> Tensor:
    """Generate voxel coordinate grid

    Args:
        shape: Shape of the grid
        device: Device of the grid

    Returns: Tensor with shape (1, len(shape), dim_1, ..., dim_{len(shape)})
    """
    axes = [
        linspace(start=0, end=int(dim_size) - 1, steps=int(dim_size), device=device, dtype=dtype)
        for dim_size in shape
    ]
    coordinates = stack(meshgrid(axes, indexing="ij"), dim=0)
    return coordinates[None]


@script
def _broadcast_batch_size(tensor_1: Tensor, tensor_2: Tensor) -> Tuple[Tensor, Tensor]:
    batch_size = max(tensor_1.size(0), tensor_2.size(0))
    if tensor_1.size(0) == 1 and batch_size != 1:
        tensor_1 = tensor_1[0].expand((batch_size,) + tensor_1.shape[1:])
    elif tensor_2.size(0) == 1 and batch_size != 1:
        tensor_2 = tensor_2[0].expand((batch_size,) + tensor_2.shape[1:])
    elif tensor_1.size(0) != tensor_2.size(0) and batch_size != 1:
        raise ValueError("Can not broadcast batch size")
    return tensor_1, tensor_2


@script
def _match_grid_shape_to_dims(grid: Tensor) -> Tensor:
    batch_size = grid.size(0)
    n_dims = grid.size(1)
    grid_shape = grid.shape[2:]
    dim_matched_grid_shape = (
        (1,) * max(0, n_dims - grid.ndim + 1) + grid_shape[: n_dims - 1] + (-1,)
    )
    return grid.view(
        (
            batch_size,
            n_dims,
        )
        + dim_matched_grid_shape
    )


@script
def interpolate(
    volume: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode: str = "border"
) -> Tensor:
    """Interpolates in voxel coordinates

    Args:
        volume: Tensor with shape
            (batch_size, [channel_1, ..., channel_n, ]dim_1, ..., dim_{n_dims})
        grid: Tensor with shape (batch_size, n_dims, *target_shape)

    Returns: Tensor with shape (batch_size, channel_1, ..., channel_n, *target_shape)
    """
    if grid.ndim == 1:
        grid = grid[None]
    n_dims = grid.size(1)
    channel_shape = volume.shape[1:-n_dims]
    volume_shape = volume.shape[-n_dims:]
    target_shape = grid.shape[2:]
    dim_matched_grid = _match_grid_shape_to_dims(grid)
    normalized_grid = convert_voxel_to_normalized_coordinates(dim_matched_grid, list(volume_shape))
    simplified_volume = volume.view((volume.size(0), -1) + volume_shape)
    permuted_volume = simplified_volume.permute(
        [0, 1] + list(range(simplified_volume.ndim - 1, 2 - 1, -1))
    )
    permuted_grid = move_channels_last(normalized_grid, 1)
    permuted_volume, permuted_grid = _broadcast_batch_size(permuted_volume, permuted_grid)
    return grid_sample(
        input=permuted_volume,
        grid=permuted_grid,
        align_corners=True,
        mode=mode,
        padding_mode=padding_mode,
    ).view((-1,) + channel_shape + target_shape)


def spline_interpolate(
    volume: Tensor,
    grid: Tensor,
    bound: int,
    prefilter: bool = False,
    degree: int = 1,
    extrapolate: bool = True,
) -> Tensor:
    """Interpolates in voxel coordinates

    Args:
        volume: Tensor with shape
            (batch_size, [channel_1, ..., channel_n, ]dim_1, ..., dim_{n_dims})
        grid: Tensor with shape ([batch_size, ]n_dims, *target_shape)

    Returns: Tensor with shape (batch_size, channel_1, ..., channel_n, *target_shape)
    """
    if grid.ndim == 1:
        grid = grid[None]
    n_dims = grid.size(1)
    channel_shape = volume.shape[1:-n_dims]
    volume_shape = volume.shape[-n_dims:]
    target_shape = grid.shape[2:]
    dim_matched_grid = _match_grid_shape_to_dims(grid)
    permuted_grid = move_channels_last(dim_matched_grid, 1)
    simplified_volume = volume.view((volume.size(0), -1) + (volume_shape))
    simplified_volume, permuted_grid = _broadcast_batch_size(simplified_volume, permuted_grid)
    # import dynamically to avoid dependency on nitorch
    nitorch_grid_pull = import_object("nitorch.spatial.grid_pull")
    return nitorch_grid_pull(
        simplified_volume,
        grid=permuted_grid,
        interpolation=degree,
        bound=bound,
        prefilter=prefilter,
        extrapolate=extrapolate,
    ).view((-1,) + channel_shape + target_shape)


@script
def integrate_svf(
    stationary_velocity_field: Tensor,
    squarings: int = 7,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> Tensor:
    """Integrate stationary velocity field in voxel coordinates"""
    grid = generate_voxel_coordinate_grid(
        shape=stationary_velocity_field.shape[2:],
        device=stationary_velocity_field.device,
        dtype=stationary_velocity_field.dtype,
    )
    integrated = stationary_velocity_field / 2**squarings
    for _ in range(squarings):
        integrated = (
            interpolate(integrated, grid=integrated + grid, mode=mode, padding_mode=padding_mode)
            + integrated
        )
    return integrated


@script
def calculate_mask_based_on_bounds(
    coordinates: Tensor,
    mask_to_update: Optional[Tensor],
    min_values: List[float],
    max_values: List[float],
    dtype: torch_dtype,
) -> Tensor:
    """Calculate mask at coordinates

    Args:
        coordinates_at_voxel_coordinates: Tensor with shape ([batch_size, ]n_dims, *target_shape)
        mask: Tensor with shape ([batch_size, ]1, *target_shape)
        min_values: Values below this are added to the mask
        max_values: Values above this are added to the mask
        dtype: Type of the generated mask
    """
    coordinates = move_channels_last(coordinates.detach())
    min_values_tensor = tensor(min_values, dtype=coordinates.dtype, device=coordinates.device)
    max_values_tensor = tensor(max_values, dtype=coordinates.dtype, device=coordinates.device)
    normalized_coordinates = (coordinates - min_values_tensor) / (
        max_values_tensor - min_values_tensor
    )
    fov_mask = (normalized_coordinates >= 0) & (normalized_coordinates <= 1)
    fov_mask = move_channels_first(torch_all(fov_mask, dim=-1, keepdim=True))
    if mask_to_update is not None:
        return fov_mask.type(dtype) * mask_to_update
    return fov_mask.type(dtype)


@script
def calculate_mask_at_voxel_coordinates(
    coordinates_at_voxel_coordinates: Tensor,
    mask_to_update: Optional[Tensor],
    volume_shape: List[int],
    dtype: torch_dtype,
) -> Optional[Tensor]:
    """Calculate mask at coordinates

    Args:
        coordinates_at_voxel_coordinates: Tensor with shape ([batch_size, ]n_dims, *target_shape)
        mask: Tensor with shape ([batch_size, ]1, *target_shape)
        volume_shape: Shape of the volume
        dtype: Type of the generated mask
    """
    return calculate_mask_based_on_bounds(
        coordinates=coordinates_at_voxel_coordinates,
        mask_to_update=mask_to_update,
        min_values=[0.0] * len(volume_shape),
        max_values=[float(dim_size - 1) for dim_size in volume_shape],
        dtype=dtype,
    )
