"""Derivative estimation based on finite difference"""

from itertools import product
from types import EllipsisType
from typing import Optional, Sequence

from torch import Tensor, stack

from util.dimension_order import index_by_channel_dims, num_spatial_dims


def estimate_spatial_derivatives(
    volume: Tensor,
    spacing: float,
    spatial_dim: int,
    n_channel_dims=1,
    other_dims: str | None = None,
    central: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    """Calculate spatial derivatives over a dimension estimated using finite differences

    Args:
        volume: Derivative is calculate over values of this mapping
            Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{n_dims})
        spatial_dim: Dimension over which to compute the derivative,
            indexing starts from 0 and corresponds to dim_1 above.
        spacing: Spacing between voxels over the spatial_dim
        n_channel_dims: Number of channel dimensions
        other_dims: If not given, the shape over other dimensions will not
                change. If given, must one of the following options:
            average: Other dimensions are averaged over two consequtive slices
                to obtain same shape difference as the dimension over which the
                derivative is computed. This can not be used if central == True.
            crop: Other dimensions are cropped to obtain same shape difference
                as the dimension over which the derivative is computed. If
                central == False, equals to the option crop_last.
            crop_first: Other dimensions are cropped to obtain same shape
                difference as the dimension over which the derivative is
                computed by cropping the first element. This can not be used if
                central == True.
            crop_last: Other dimensions are cropped to obtain same shape
                difference as the dimension over which the derivative is
                computed by cropping the last element. This can not be used if
                central == True.
            crop_both: Apply both crop_first and crop_last to all dimensions and
                include all variants as a separate dimension with size
                2**n_dims. Note that the spatial dimension is not cropped and
                hence the channel contains the same elements repeated twice.
                Mainly useful when using through
                estimate_spatial_jacobian_matrices as it then corresponds to
                calculating the Jacobian matrix at each possible grid corner
                assuming linear interpolation.
        central: Whether to use central difference [f(x + 1)  - f(x - 1)] / 2 or not
            f(x + 1) - f(x)
        out: Save output to this Tensor

    Returns:
        if central and other_dims == "crop": Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1 - 2, ..., dim_{n_dims} - 2)
        elif central and other_dims is None: Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{spatial_dim} - 2, ..., dim_{n_dims})
        elif not central and other_dims not in (None, "crop_both"): Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1 - 1, ..., dim_{n_dims} - 1)
        elif not central and other_dims == "crop_both": Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            2**n_dims, dim_1 - 1, ..., dim_{n_dims} - 1)
        elif not central and other_dims is None: Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{spatial_dim} - 1, ..., dim_{n_dims})
    """
    if central and other_dims not in (None, "crop"):
        raise ValueError(f'Can not use central difference with option other_dims == "{other_dims}"')
    n_spatial_dims = num_spatial_dims(volume.ndim, n_channel_dims)
    if other_dims == "crop":
        other_crop = slice(1, -1) if central else slice(None, -1)
    elif other_dims == "crop_first":
        other_crop = slice(1, None)
    elif other_dims == "crop_last":
        other_crop = slice(None, -1)
    else:
        other_crop = slice(None)
    if central:
        front_crop = slice(2, None)
        back_crop = slice(None, -2)
    else:
        front_crop = slice(1, None)
        back_crop = slice(None, -1)
    front_cropping_slice = (...,) + tuple(
        front_crop if i == spatial_dim else other_crop for i in range(n_spatial_dims)
    )
    back_cropping_slice = (...,) + tuple(
        back_crop if i == spatial_dim else other_crop for i in range(n_spatial_dims)
    )
    derivatives = (volume[front_cropping_slice] - volume[back_cropping_slice]) / spacing
    if central:
        derivatives = derivatives / 2
    if other_dims == "average":
        front_cropping_slice_other_dims = (...,) + tuple(
            slice(None) if i == spatial_dim else slice(1, None) for i in range(n_spatial_dims)
        )
        back_cropping_slice_other_dims = (...,) + tuple(
            slice(None) if i == spatial_dim else slice(None, -1) for i in range(n_spatial_dims)
        )
        derivatives = (
            derivatives[front_cropping_slice_other_dims]
            + derivatives[back_cropping_slice_other_dims]
        ) / 2
    if other_dims == "crop_both":
        last_channel_dim = index_by_channel_dims(
            volume.ndim, channel_dim_index=n_channel_dims - 1, n_channel_dims=n_channel_dims
        )
        if out is not None:
            for index, shifts in enumerate(
                product([slice(None, -1), slice(1, None)], repeat=n_spatial_dims)
            ):
                out[(slice(None),) * (last_channel_dim + 1) + (index,)] = derivatives[
                    _shifting_slice(spatial_dim=spatial_dim, shifts=shifts)
                ]
            return out
        return stack(
            [
                derivatives[_shifting_slice(spatial_dim=spatial_dim, shifts=shifts)]
                for shifts in product([slice(None, -1), slice(1, None)], repeat=n_spatial_dims)
            ],
            dim=last_channel_dim + 1,
        )
    if out is not None:
        out[:] = derivatives
        return out
    return derivatives


def estimate_spatial_jacobian_matrices(
    volume: Tensor,
    spacing: Optional[Sequence[float]] = None,
    n_channel_dims: int = 1,
    other_dims: str = "average",
    central: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    """Calculate local Jacobian matrices of a volume estimated using finite differences

    Args:
        values: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims}),
            regularly sampled values of some mapping with the given spacing
        spacing: Voxel sizes along each dimension
        n_channel_dims: Number of channel dimensions
        other_dims: See option other_dims of algorithm.finite_difference.
        central: See option central of algorithm.finite_difference

    Returns:
        if central
            Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            n_dims, dim_1 - 2, ..., dim_{n_dims} - 2)
        else:
            if other_dims != "crop_both":
                Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
                    n_dims, dim_1 - 1, ..., dim_{n_dims} - 1)
            else:
                Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
                    n_dims, 2**n_dims, dim_1 - 1, ..., dim_{n_dims} - 1)
    """
    if central and other_dims == "crop_both":
        raise ValueError(f'Can not use central difference with option other_dims == "{other_dims}"')
    n_spatial_dims = num_spatial_dims(volume.ndim, n_channel_dims)
    if spacing is None:
        spacing = [1.0] * n_spatial_dims
    last_channel_dim = index_by_channel_dims(
        volume.ndim, channel_dim_index=n_channel_dims - 1, n_channel_dims=n_channel_dims
    )
    if out is not None:
        for dim in range(n_spatial_dims):
            estimate_spatial_derivatives(
                volume=volume,
                spatial_dim=dim,
                spacing=float(spacing[dim]),
                n_channel_dims=n_channel_dims,
                other_dims=other_dims,
                central=central,
                out=out[(slice(None),) * (last_channel_dim + 1) + (dim,)],
            )
        return out
    return stack(
        [
            estimate_spatial_derivatives(
                volume=volume,
                spatial_dim=dim,
                spacing=float(spacing[dim]),
                n_channel_dims=n_channel_dims,
                other_dims=other_dims,
                central=central,
            )
            for dim in range(n_spatial_dims)
        ],
        dim=last_channel_dim + 1,
    )


def _shifting_slice(spatial_dim: int, shifts: Sequence[slice]) -> tuple[slice | EllipsisType, ...]:
    return (...,) + tuple(
        slice(None) if spatial_dim == dim else shifts[dim] for dim in range(len(shifts))
    )
