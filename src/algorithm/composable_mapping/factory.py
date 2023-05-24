"""Factory methods for generating useful composable mappings"""

from typing import Optional, Sequence

from torch import Tensor
from torch import device as torch_device
from torch import diag
from torch import dtype as torch_dtype
from torch import tensor

from ..affine_transformation import (
    AffineTransformationTypeDefinition,
    generate_affine_transformation_matrix,
)
from .affine import Affine, ComposableAffine, CPUComposableAffine
from .grid_mapping import GridCoordinateMapping, GridMappingArgs, GridVolume, as_displacement_field
from .identity import ComposableIdentity
from .interface import IComposableMapping, VoxelCoordinateSystem
from .masked_tensor import VoxelCoordinateGrid


class ComposableFactory:
    """Class acting as a namespace for methods for generating composable mappings"""

    @staticmethod
    def create_identity() -> IComposableMapping:
        """Create identity mapping"""
        return ComposableIdentity()

    @staticmethod
    def create_affine(affine_transformation: Tensor) -> IComposableMapping:
        """Create affine mapping"""
        return ComposableAffine(Affine(affine_transformation))

    @staticmethod
    def create_affine_from_parameters(
        parameters: Tensor, transformation_type: AffineTransformationTypeDefinition
    ) -> IComposableMapping:
        """Create affine mapping"""
        return ComposableAffine(
            Affine(generate_affine_transformation_matrix(parameters, transformation_type))
        )

    @staticmethod
    def create_volume(
        data: Tensor,
        coordinate_system: VoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        grid_volume = GridVolume(
            data=data, grid_mapping_args=grid_mapping_args, n_channel_dims=n_channel_dims, mask=mask
        )
        return grid_volume.compose(coordinate_system.to_voxel_coordinates)

    @staticmethod
    def create_dense_mapping(
        displacement_field: Tensor,
        coordinate_system: VoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create mapping based on dense displacement field"""
        grid_volume = GridCoordinateMapping(
            displacement_field=displacement_field, grid_mapping_args=grid_mapping_args, mask=mask
        )
        return coordinate_system.from_voxel_coordinates.compose(grid_volume).compose(
            coordinate_system.to_voxel_coordinates
        )

    @classmethod
    def resample_to_volume(
        cls,
        mapping: IComposableMapping,
        coordinate_system: VoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        n_channel_dims: int = 1,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        samples = mapping(coordinate_system.grid)
        return cls.create_volume(
            data=samples.generate_values(device=device, dtype=dtype),
            coordinate_system=coordinate_system,
            grid_mapping_args=grid_mapping_args,
            n_channel_dims=n_channel_dims,
            mask=samples.mask,
        )

    @classmethod
    def resample_to_dense_mapping(
        cls,
        mapping: IComposableMapping,
        coordinate_system: VoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
        clear_mask: bool = False,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        displacement_field, mask = as_displacement_field(
            mapping=mapping, coordinate_system=coordinate_system, device=device, dtype=dtype
        )
        return cls.create_dense_mapping(
            displacement_field=displacement_field,
            coordinate_system=coordinate_system,
            grid_mapping_args=grid_mapping_args,
            mask=None if clear_mask else mask,
        )


class CoordinateSystemFactory:
    """Namespace for coordinate system generation functions"""

    @classmethod
    def centered_normalized(
        cls,
        original_grid_shape: Sequence[int],
        voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
        dtype: torch_dtype | None = None,
    ) -> VoxelCoordinateSystem:
        """Create normalized coordinate system

        Origin is in the middle of the voxel space and voxels are assumed to be
        squares with the sampled value in the middle.

        Coordinates are scaled such that the whole FOV fits inside values from
        -1 to 1 for the hypothetical original grid assuming downsampling factor
        1.0. The actual grid then has voxel size of the original grid divided by
        the downsampling factor and is located in the middle with respect to the
        the original grid.

        Arguments:
            original_grid_shape: Shape of the hypothetical grid
            grid_shape: Shape of the actual grid, defaults to original_grid_shape
            voxel_size: Voxel size of the coordinate grid, only relative
                differences have meaning
            downsampling_factor: Downsampling factor of the actual grid compared
                to the original grid, defaults to no scaling.
        """
        voxel_size, grid_shape, downsampling_factor = cls._handle_normalized_optional_inputs(
            original_grid_shape=original_grid_shape,
            voxel_size=voxel_size,
            grid_shape=grid_shape,
            downsampling_factor=downsampling_factor,
        )
        world_origin_in_voxels = [(dim_size - 1) / 2 for dim_size in grid_shape]
        world_to_voxel_scale = cls._normalized_scale(
            voxel_size=voxel_size,
            grid_shape=original_grid_shape,
            downsampling_factor=downsampling_factor,
        )
        return cls._generate_coordinate_system(
            grid_shape=grid_shape,
            world_to_voxel_scale=world_to_voxel_scale,
            world_origin_in_voxels=world_origin_in_voxels,
            dtype=dtype,
        )

    @classmethod
    def top_left_aligned_normalized(
        cls,
        original_grid_shape: Sequence[int],
        voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
        dtype: torch_dtype | None = None,
    ) -> VoxelCoordinateSystem:
        """Create normalized coordinate system

        Voxels are assumed to be squares with the sampled value in the middle.

        Coordinates are scaled such that the whole FOV fits inside values from
        -1 to 1 for the hypothetical original grid assuming downsampling factor
        1.0. Original grid is centered. The actual grid then has voxel size of
        the original grid divided by the downsampling factor and top-left corner
        is aligned with the original grid.

        Arguments:
            original_grid_shape: Shape of the hypothetical grid
            grid_shape: Shape of the actual grid, defaults to original_grid_shape
            voxel_size: Voxel size of the coordinate grid, only relative
                differences have meaning
            downsampling_factor: Downsampling factor of the actual grid compared
                to the original grid, defaults to no scaling.
        """
        voxel_size, grid_shape, downsampling_factor = cls._handle_normalized_optional_inputs(
            original_grid_shape=original_grid_shape,
            voxel_size=voxel_size,
            grid_shape=grid_shape,
            downsampling_factor=downsampling_factor,
        )
        world_to_voxel_scale = cls._normalized_scale(
            voxel_size=voxel_size,
            grid_shape=original_grid_shape,
            downsampling_factor=downsampling_factor,
        )
        world_origin_in_voxels = [
            (dim_size / dim_downsampling_factor - 1) / 2
            for (dim_downsampling_factor, dim_size) in zip(downsampling_factor, original_grid_shape)
        ]
        return cls._generate_coordinate_system(
            grid_shape=grid_shape,
            world_to_voxel_scale=world_to_voxel_scale,
            world_origin_in_voxels=world_origin_in_voxels,
            dtype=dtype,
        )

    @classmethod
    def voxel(
        cls,
        grid_shape: Sequence[int],
        voxel_size: Optional[Sequence[float]] = None,
        dtype: torch_dtype | None = None,
    ) -> VoxelCoordinateSystem:
        """Create voxel coordinate system"""
        if voxel_size is None:
            voxel_size = [1.0] * len(grid_shape)
        return cls._generate_coordinate_system(
            grid_shape=grid_shape,
            world_to_voxel_scale=[1 / dim_voxel_size for dim_voxel_size in voxel_size],
            world_origin_in_voxels=[0.0] * len(grid_shape),
            dtype=dtype,
        )

    @staticmethod
    def _handle_normalized_optional_inputs(
        original_grid_shape: Sequence[int],
        voxel_size: Optional[Sequence[float]],
        grid_shape: Optional[Sequence[int]],
        downsampling_factor: Optional[Sequence[float]],
    ) -> tuple[Sequence[float], Sequence[int], Sequence[float]]:
        n_dims = len(original_grid_shape)
        if voxel_size is None:
            voxel_size = [1.0] * n_dims
        if grid_shape is None:
            grid_shape = original_grid_shape
        if downsampling_factor is None:
            downsampling_factor = [1.0] * n_dims
        return voxel_size, grid_shape, downsampling_factor

    @classmethod
    def _generate_coordinate_system(
        cls,
        grid_shape: Sequence[int],
        world_to_voxel_scale: Sequence[float],
        world_origin_in_voxels: Sequence[float],
        dtype: torch_dtype | None,
    ) -> VoxelCoordinateSystem:
        voxel_to_world_scale = [1 / dim_voxel_size for dim_voxel_size in world_to_voxel_scale]
        transformation_matrix = cls._generate_scale_and_translation_matrix(
            scale=world_to_voxel_scale, translation=world_origin_in_voxels, dtype=dtype
        )
        to_voxel_coordinates_affine = CPUComposableAffine(transformation_matrix).cache()
        from_voxel_coordinates_affine = to_voxel_coordinates_affine.invert().cache()
        to_voxel_coordinates = ComposableAffine(to_voxel_coordinates_affine)
        from_voxel_coordinates = ComposableAffine(from_voxel_coordinates_affine)
        voxel_grid = VoxelCoordinateGrid(grid_shape)
        return VoxelCoordinateSystem(
            from_voxel_coordinates=from_voxel_coordinates,
            to_voxel_coordinates=to_voxel_coordinates,
            grid=from_voxel_coordinates(voxel_grid),
            voxel_grid=voxel_grid,
            grid_spacing=voxel_to_world_scale,
        )

    @staticmethod
    def _normalized_scale(
        voxel_size: Sequence[float], grid_shape: Sequence[int], downsampling_factor: Sequence[float]
    ) -> Sequence[float]:
        """Scaling applied to normalized coordinate systems

        Scaling is computed such that the whole volume just fits inside
        values [-1, 1].
        """
        scale = (
            max(
                dim_size * dim_voxel_size
                for (dim_voxel_size, dim_size) in zip(voxel_size, grid_shape)
            )
            / 2
        )
        world_to_voxel_scale = [
            scale / dim_voxel_size / dim_downsampling_factor
            for (dim_voxel_size, dim_downsampling_factor) in zip(voxel_size, downsampling_factor)
        ]
        return world_to_voxel_scale

    @staticmethod
    def _generate_scale_and_translation_matrix(
        scale: Sequence[float], translation: Sequence[float], dtype: torch_dtype | None
    ) -> Tensor:
        matrix = diag(tensor(list(scale) + [1.0], dtype=dtype))
        matrix[:-1, -1] = tensor(translation, dtype=dtype)
        return matrix
