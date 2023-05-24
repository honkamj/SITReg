"""AIR-filetype related implementations http://air.bmap.ucla.edu/AIR5/"""

from ctypes import Structure, c_char, c_double, c_int, c_uint, c_ushort
from io import BufferedIOBase, RawIOBase
from typing import Sequence

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import tensor

from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs

AIR_CONFIG_MAX_PATH_LENGTH = 128


AIR_CONFIG_MAX_COMMENT_LENGTH = 128


AIR_CONFIG_RESERVED_LENGTH = 116


class AIRKeyinfo(Structure):
    """Represents air key info struct"""

    _fields_ = [
        ("bits", c_uint),
        ("x_dim", c_uint),
        ("y_dim", c_uint),
        ("z_dim", c_uint),
        ("x_size", c_double),
        ("y_size", c_double),
        ("z_size", c_double),
    ]


class AIR16(Structure):
    """Represents air16 struct

    Data in .air-files correspond to the struct
    """

    _fields_ = [
        ("e", c_double * 4 * 4),
        ("s_file", c_char * AIR_CONFIG_MAX_PATH_LENGTH),
        ("s", AIRKeyinfo),
        ("r_file", c_char * AIR_CONFIG_MAX_PATH_LENGTH),
        ("r", AIRKeyinfo),
        ("comment", c_char * AIR_CONFIG_MAX_COMMENT_LENGTH),
        ("s_hash", c_int),
        ("r_hash", c_int),
        ("s_volume", c_ushort),
        ("r_volume", c_ushort),
        ("reserved", c_char * AIR_CONFIG_RESERVED_LENGTH),
    ]


def read_air_file(file: BufferedIOBase | RawIOBase) -> AIR16:
    """Read air header file"""
    output = AIR16()
    file.readinto(output)
    if file.readinto(output) != 0:
        raise RuntimeError("The file does not conform to the implemented AIR datatype.")
    return output


def get_transformation_matrix(
    air: AIR16, device: torch_device | None = None, dtype: torch_dtype | None = None
) -> Tensor:
    """Get voxel coordinate transformation matrix of the air transformation as Tensor"""
    return tensor(air.e, device=device, dtype=dtype).T


def _get_smallest_voxel_width(air: AIR16) -> float:
    return min(air.s.x_size, air.s.y_size, air.s.z_size)


def _get_zooms(air: AIR16) -> tuple[float, float, float]:
    smallest_voxel_width = _get_smallest_voxel_width(air)
    return (
        air.s.x_size / smallest_voxel_width,
        air.s.y_size / smallest_voxel_width,
        air.s.z_size / smallest_voxel_width,
    )


def get_source_shape(air: AIR16) -> tuple[int, int, int]:
    """Get source shape of the air transformation"""
    return (air.r.x_dim, air.r.y_dim, air.r.z_dim)


def get_source_voxel_size(air: AIR16) -> tuple[float, float, float]:
    """Get source voxel size of the air transformation"""
    return (air.r.x_size, air.r.y_size, air.r.z_size)


def get_transformed_shape(air: AIR16) -> tuple[int, int, int]:
    """Get after transformation shape of the air transformation"""
    zooms = _get_zooms(air)
    return (
        int((air.s.x_dim - 1) * zooms[0] + 1),
        int((air.s.y_dim - 1) * zooms[1] + 1),
        int((air.s.z_dim - 1) * zooms[2] + 1),
    )


def get_transformed_voxel_size(air: AIR16) -> tuple[float, float, float]:
    """Get after transformation voxel size of the air transformation"""
    return (_get_smallest_voxel_width(air),) * 3


def transform_volume_by_air(
    air: AIR16,
    volume: Tensor,
    target_voxel_size: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    grid_mapping_args: GridMappingArgs,
) -> Tensor:
    """Transform volume by air transform

    Args:
        air: Air transform
        volume: Tensor with shape ([batch_size, ]n_channels, dim_1, dim_2, dim_3)
        target_voxel_size: Voxel size of the target space where the volume will be transformed
        target_shape: Target volume shape
        grid_mapping_args: Defines how to interpolate

    Assumes that the shape of the volume and source space of the air tranformation match.
    """
    source_coordinate_system = CoordinateSystemFactory.voxel(
        get_source_shape(air), voxel_size=get_source_voxel_size(air), dtype=volume.dtype
    )
    transformation_target_coordinate_system = CoordinateSystemFactory.voxel(
        get_transformed_shape(air), voxel_size=get_transformed_voxel_size(air), dtype=volume.dtype
    )
    target_coordinate_system = CoordinateSystemFactory.voxel(
        target_shape, voxel_size=target_voxel_size, dtype=volume.dtype
    )
    volume_mapping = ComposableFactory.create_volume(
        volume,
        coordinate_system=source_coordinate_system,
        grid_mapping_args=grid_mapping_args,
    )
    voxel_coordinate_transformation = ComposableFactory.create_affine(
        get_transformation_matrix(air, device=volume.device, dtype=volume.dtype)
    )
    return (
        volume_mapping.compose(source_coordinate_system.from_voxel_coordinates)
        .compose(voxel_coordinate_transformation)
        .compose(transformation_target_coordinate_system.to_voxel_coordinates)(
            target_coordinate_system.grid
        )
        .generate_values()
    )


def inverse_transform_volume_by_air(
    air: AIR16,
    volume: Tensor,
    target_voxel_size: Sequence[float],
    grid_mapping_args: GridMappingArgs,
) -> Tensor:
    """Transform volume by air transform to the source space of the transform

    Args:
        air: Air transform
        volume: Tensor with shape ([batch_size, ]n_channels, dim_1, dim_2, dim_3)
        target_voxel_size: Voxel size of the target space where the volume is located
        grid_mapping_args: Defines how to interpolate

    Assumes that the shape of the volume and source space of the air tranformation match.
    """
    source_coordinate_system = CoordinateSystemFactory.voxel(
        get_source_shape(air), voxel_size=get_source_voxel_size(air), dtype=volume.dtype
    )
    transformation_target_coordinate_system = CoordinateSystemFactory.voxel(
        get_transformed_shape(air), voxel_size=get_transformed_voxel_size(air), dtype=volume.dtype
    )
    target_coordinate_system = CoordinateSystemFactory.voxel(
        volume.shape[-3:], voxel_size=target_voxel_size, dtype=volume.dtype
    )
    volume_mapping = ComposableFactory.create_volume(
        volume,
        coordinate_system=target_coordinate_system,
        grid_mapping_args=grid_mapping_args,
    )
    voxel_coordinate_transformation = ComposableFactory.create_affine(
        get_transformation_matrix(air, device=volume.device, dtype=volume.dtype)
    )
    return volume_mapping.compose(
        transformation_target_coordinate_system.from_voxel_coordinates
    ).compose(
        voxel_coordinate_transformation.invert()
    ).compose(
        source_coordinate_system.to_voxel_coordinates
    )(source_coordinate_system.grid).generate_values()
