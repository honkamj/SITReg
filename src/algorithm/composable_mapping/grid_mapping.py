"""Grid based mappings"""

from typing import Optional

from attr import define, field
from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from algorithm.composable_mapping.interface import IComposableMapping

from ..dense_deformation import calculate_mask_at_voxel_coordinates
from ..interface import IInterpolator
from .base import BaseComposableMapping
from .interface import (
    IComposableMapping,
    IMaskedTensor,
    IRegularGridTensor,
    VoxelCoordinateSystem,
)
from .masked_tensor import MaskedTensor


@define
class GridMappingArgs:
    """Represents arguments for creating grid volume

    Arguments:
        interpolator: Interpolator which interpolates the volume at voxel coordinates
        mask_interpolator: Interpolator which interpolates the mask at voxel coordinates,
            defaults to interpolator
        mask_outside_fov: Whether to update interpolation locations outside field of view
            to the mask
        mask_threshold: All values under threshold are set to zero and above it to one,
            if None, no thresholding will be done
    """

    interpolator: IInterpolator
    mask_interpolator: IInterpolator = field()

    @mask_interpolator.default
    def _default_mask_interpolator(self) -> IInterpolator:
        return self.interpolator

    mask_outside_fov: bool = True
    mask_threshold: Optional[float] = 1.0 - 1e-5


class GridVolume(BaseComposableMapping):
    """Continuously defined volume on voxel coordinates based on
    and interpolation/extrapolation method

    Arguments:
        data: Tensor with shape (batch_size, *channel_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
        n_channel_dims: Number of channel dimensions in data
    """

    def __init__(
        self,
        data: Tensor,
        grid_mapping_args: GridMappingArgs,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
    ) -> None:
        self._data = data
        self._mask = mask
        self._grid_mapping_args = grid_mapping_args
        self._n_channel_dims = n_channel_dims
        self._volume_shape = data.shape[n_channel_dims + 1 :]
        self._n_dims = len(self._volume_shape)
        if mask is not None and mask.device != data.device:
            raise RuntimeError("Devices do not match")

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self._evaluate(masked_coordinates)[0]

    def _evaluate(
        self, masked_coordinates: IMaskedTensor
    ) -> tuple[IMaskedTensor, Optional[Tensor]]:
        if isinstance(masked_coordinates, IRegularGridTensor):
            return self._evaluate_grid(masked_coordinates)
        return self._evaluate_and_return_coordinates(masked_coordinates)

    def _evaluate_grid(
        self, masked_coordinates: IRegularGridTensor
    ) -> tuple[IMaskedTensor, Optional[Tensor]]:
        target_slice = masked_coordinates.reduce_to_slice(self._volume_shape)
        if target_slice is None:
            return self._evaluate_and_return_coordinates(masked_coordinates)
        if self._mask is not None:
            mask: Optional[Tensor] = self._mask[target_slice]
        else:
            mask = None
        return MaskedTensor(self._data[target_slice], mask, self._n_channel_dims), None

    def _evaluate_and_return_coordinates(
        self, masked_coordinates: IMaskedTensor
    ) -> tuple[IMaskedTensor, Tensor]:
        voxel_coordinates = masked_coordinates.generate_values(
            device=self._data.device, dtype=self._data.dtype
        )
        values = self._interpolate(voxel_coordinates)
        mask = self._update_mask(self._mask, voxel_coordinates, masked_coordinates.mask)
        return MaskedTensor(values, mask, self._n_channel_dims), voxel_coordinates

    def _interpolate(self, voxel_coordinates: Tensor) -> Tensor:
        values = self._grid_mapping_args.interpolator(self._data, voxel_coordinates)
        return values

    def _update_mask(
        self,
        mask: Tensor | None,
        voxel_coordinates: Tensor,
        voxel_coordinates_mask: Tensor | None,
    ) -> Optional[Tensor]:
        if mask is not None:
            mask = self._grid_mapping_args.mask_interpolator(mask, voxel_coordinates)
            mask = self._threshold_mask(mask)
        if self._grid_mapping_args.mask_outside_fov:
            mask = calculate_mask_at_voxel_coordinates(
                voxel_coordinates,
                mask_to_update=mask,
                volume_shape=self._volume_shape,
                dtype=self._data.dtype,
            )
        if voxel_coordinates_mask is not None:
            if mask is None:
                mask = voxel_coordinates_mask
            else:
                mask = voxel_coordinates_mask * mask
        return mask

    def _threshold_mask(self, mask: Tensor) -> Tensor:
        if self._grid_mapping_args.mask_threshold is not None:
            return (
                (mask < self._grid_mapping_args.mask_threshold)
                .logical_not()
                .type(mask.dtype)
            )
        return mask

    def invert(self, **kwargs) -> IComposableMapping:
        raise NotImplementedError("No inversion for generic volumes")

    def detach(self) -> "GridVolume":
        return GridVolume(
            data=self._data.detach(),
            grid_mapping_args=self._grid_mapping_args,
            n_channel_dims=self._n_channel_dims,
            mask=self._mask.detach() if self._mask is not None else None,
        )

    def to_dtype(self, dtype: torch_dtype) -> "GridVolume":
        return GridVolume(
            data=self._data.type(dtype),
            grid_mapping_args=self._grid_mapping_args,
            n_channel_dims=self._n_channel_dims,
            mask=self._mask.type(dtype) if self._mask is not None else None,
        )

    def to_device(self, device: torch_device) -> "GridVolume":
        return GridVolume(
            data=self._data.to(device=device),
            grid_mapping_args=self._grid_mapping_args,
            n_channel_dims=self._n_channel_dims,
            mask=self._mask.to(device=device) if self._mask is not None else None,
        )


class GridCoordinateMapping(GridVolume):
    """Continuously defined mapping based on regular grid samples

    Arguments:
        displacement_field: Displacement field in voxel coordinates, Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
    """

    def __init__(
        self,
        displacement_field: Tensor,
        grid_mapping_args: GridMappingArgs,
        mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            data=displacement_field,
            grid_mapping_args=grid_mapping_args,
            mask=mask,
            n_channel_dims=1,
        )

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        displacement_field_values, voxel_coordinates = super()._evaluate(
            masked_coordinates
        )
        if voxel_coordinates is None:
            voxel_coordinates = masked_coordinates.generate_values(
                device=self._data.device, dtype=self._data.dtype
            )
        return MaskedTensor(
            values=voxel_coordinates
            + displacement_field_values.generate_values(
                device=self._data.device, dtype=self._data.dtype
            ),
            mask=displacement_field_values.mask,
        )

    def invert(self, **inversion_parameters) -> "_GridCoordinateMappingInverse":
        """Fixed point invert displacement field

        inversion_parameters:
            fixed_point_inversion_arguments (Mapping[str, Any]): Arguments for
                fixed point inversion
        """
        fixed_point_inversion_arguments = inversion_parameters.get(
            "fixed_point_inversion_arguments", {}
        )
        deformation_inversion_arguments = DeformationInversionArguments(
            interpolator=self._grid_mapping_args.interpolator,
            forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
            backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
            forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
            backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
        )
        return _GridCoordinateMappingInverse(
            displacement_field=self._data,
            grid_mapping_args=self._grid_mapping_args,
            inversion_arguments=deformation_inversion_arguments,
            mask=self._mask,
        )

    def detach(self) -> "GridCoordinateMapping":
        return GridCoordinateMapping(
            displacement_field=self._data.detach(),
            grid_mapping_args=self._grid_mapping_args,
            mask=self._mask.detach() if self._mask is not None else None,
        )

    def to_dtype(self, dtype: torch_dtype) -> "GridCoordinateMapping":
        return GridCoordinateMapping(
            displacement_field=self._data.type(dtype),
            grid_mapping_args=self._grid_mapping_args,
            mask=self._mask.type(dtype) if self._mask is not None else None,
        )

    def to_device(self, device: torch_device) -> "GridCoordinateMapping":
        return GridCoordinateMapping(
            displacement_field=self._data.to(device=device),
            grid_mapping_args=self._grid_mapping_args,
            mask=self._mask.to(device=device) if self._mask is not None else None,
        )


class _GridCoordinateMappingInverse(GridVolume):
    """Continuously defined mapping based on regular grid samples

    Arguments:
        displacement_field: Displacement field in voxel coordinates, Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
    """

    def __init__(
        self,
        displacement_field: Tensor,
        grid_mapping_args: GridMappingArgs,
        inversion_arguments: DeformationInversionArguments,
        mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            data=displacement_field,
            grid_mapping_args=grid_mapping_args,
            mask=mask,
            n_channel_dims=1,
        )
        self._inversion_arguments = inversion_arguments

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        displacement_field_values, voxel_coordinates = super()._evaluate(
            masked_coordinates
        )
        if voxel_coordinates is None:
            voxel_coordinates = masked_coordinates.generate_values(
                device=self._data.device, dtype=self._data.dtype
            )
        return MaskedTensor(
            values=voxel_coordinates
            + displacement_field_values.generate_values(
                device=self._data.device, dtype=self._data.dtype
            ),
            mask=displacement_field_values.mask,
        )

    def _evaluate_grid(
        self, masked_coordinates: IRegularGridTensor
    ) -> tuple[IMaskedTensor, Tensor]:
        target_slice = masked_coordinates.reduce_to_slice(self._volume_shape)
        if target_slice is None:
            return self._evaluate_and_return_coordinates(masked_coordinates)
        if self._mask is not None:
            mask: Optional[Tensor] = self._mask[target_slice]
        else:
            mask = None
        initial_guess = -self._data[target_slice]
        voxel_coordinates = masked_coordinates.generate_values(
            device=self._data.device, dtype=self._data.dtype
        )
        inverted = fixed_point_invert_deformation(
            displacement_field=self._data,
            arguments=self._inversion_arguments,
            initial_guess=initial_guess,
            coordinates=voxel_coordinates,
        )
        mask = self._update_inverse_mask(mask, inverted)
        return MaskedTensor(inverted, mask, self._n_channel_dims), voxel_coordinates

    def _evaluate_and_return_coordinates(
        self, masked_coordinates: IMaskedTensor
    ) -> tuple[IMaskedTensor, Tensor]:
        voxel_coordinates = masked_coordinates.generate_values(
            device=self._data.device, dtype=self._data.dtype
        )
        mask = self._update_mask(self._mask, voxel_coordinates, masked_coordinates.mask)
        inverted = fixed_point_invert_deformation(
            displacement_field=self._data,
            arguments=self._inversion_arguments,
            initial_guess=None,
            coordinates=voxel_coordinates,
        )
        mask = self._update_inverse_mask(mask, inverted)
        return MaskedTensor(inverted, mask, self._n_channel_dims), voxel_coordinates

    def _update_inverse_mask(
        self, mask: Tensor | None, inverted_coordinates: Tensor
    ) -> Tensor | None:
        if self._grid_mapping_args.mask_outside_fov:
            mask = calculate_mask_at_voxel_coordinates(
                inverted_coordinates,
                mask,
                self._volume_shape,
                dtype=inverted_coordinates.dtype,
            )
        return mask

    def invert(self, **_inversion_parameters) -> "GridCoordinateMapping":
        """Invert inverse"""
        return GridCoordinateMapping(
            displacement_field=self._data,
            grid_mapping_args=self._grid_mapping_args,
            mask=self._mask,
        )

    def detach(self) -> "_GridCoordinateMappingInverse":
        return _GridCoordinateMappingInverse(
            displacement_field=self._data.detach(),
            grid_mapping_args=self._grid_mapping_args,
            inversion_arguments=self._inversion_arguments,
            mask=self._mask.detach() if self._mask is not None else None,
        )

    def to_dtype(self, dtype: torch_dtype) -> "_GridCoordinateMappingInverse":
        return _GridCoordinateMappingInverse(
            displacement_field=self._data.type(dtype),
            grid_mapping_args=self._grid_mapping_args,
            inversion_arguments=self._inversion_arguments,
            mask=self._mask.type(dtype) if self._mask is not None else None,
        )

    def to_device(self, device: torch_device) -> "_GridCoordinateMappingInverse":
        return _GridCoordinateMappingInverse(
            displacement_field=self._data.to(device=device),
            grid_mapping_args=self._grid_mapping_args,
            inversion_arguments=self._inversion_arguments,
            mask=self._mask.to(device=device) if self._mask is not None else None,
        )


def as_displacement_field(
    mapping: IComposableMapping,
    coordinate_system: VoxelCoordinateSystem,
    device: Optional[torch_device] = None,
    dtype: Optional[torch_dtype] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """Extract displacement field from a mapping"""
    voxel_coordinate_mapping = coordinate_system.to_voxel_coordinates(
        mapping(coordinate_system.grid)
    )
    voxel_coordinate_mapping_values = voxel_coordinate_mapping.generate_values(
        device=device, dtype=dtype
    )
    displacement_field = (
        voxel_coordinate_mapping_values
        - coordinate_system.voxel_grid.generate_values(
            device=voxel_coordinate_mapping_values.device, dtype=dtype
        )
    )
    return displacement_field, voxel_coordinate_mapping.mask
