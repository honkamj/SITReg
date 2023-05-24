"""Affine transformation related algorithms"""


from math import sqrt
from typing import List, Optional

from torch import Tensor, cat, diag_embed, eye, matmul
from torch import matrix_exp as torch_matrix_exp
from torch import ones, tril_indices, triu_indices, zeros
from torch.jit import script

from util.dimension_order import (
    merge_batch_dimensions,
    move_channels_first,
    move_channels_last,
    unmerge_batch_dimensions,
)


@script
class AffineTransformationTypeDefinition:
    """Affine transformation type definition

    Corresponds to the parametrization:
    Kaji, Shizuo, and Hiroyuki Ochiai. "A concise parametrization of affine transformation." (2016)

    Arguments:
        translation: Translation included
        rotation: Rotation included
        scale: Scale included
        shear: Shear included
    """

    def __init__(self, translation: bool, rotation: bool, scale: bool, shear: bool) -> None:
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.shear = shear

    @staticmethod
    def full():
        """Generate full affine transformation"""
        return AffineTransformationTypeDefinition(True, True, True, True)

    @staticmethod
    def only_translation():
        """Generate translation only transformation"""
        return AffineTransformationTypeDefinition(True, False, False, False)

    @staticmethod
    def only_rotation():
        """Generate rotation only transformation"""
        return AffineTransformationTypeDefinition(False, True, False, False)

    @staticmethod
    def only_scale():
        """Generate shear only transformation"""
        return AffineTransformationTypeDefinition(False, False, True, False)

    @staticmethod
    def only_shear():
        """Generate shear only transformation"""
        return AffineTransformationTypeDefinition(False, False, False, True)


@script
def calculate_n_parameters(
    n_dims: int, transformation_type: AffineTransformationTypeDefinition
) -> int:
    """Calculate number of parameters from number of dims

    Args:
        n_dims: Number of dimensions
    """
    return int(
        int(transformation_type.translation) * n_dims
        + int(transformation_type.rotation) * (n_dims**2 - n_dims) / 2
        + int(transformation_type.scale) * n_dims
        + int(transformation_type.shear) * (n_dims**2 - n_dims) / 2
    )


@script
def calculate_n_dims(
    n_parameters: int, transformation_type: AffineTransformationTypeDefinition
) -> int:
    """Calculate number of dimensions from number of parameters

    Args:
        n_parameters: Number of parameters
    """
    if transformation_type.rotation or transformation_type.shear:
        square_coefficient = int(transformation_type.rotation) + int(transformation_type.shear)
        linear_coffecient = (
            2 * int(transformation_type.translation)
            - int(transformation_type.rotation)
            + 2 * int(transformation_type.scale)
            - int(transformation_type.shear)
        )
        n_dims = (
            -linear_coffecient
            + sqrt(linear_coffecient**2 + 8 * square_coefficient * n_parameters)
        ) / (2 * square_coefficient)
    elif transformation_type.translation or transformation_type.scale:
        n_dims = n_parameters / (
            int(transformation_type.translation) + int(transformation_type.scale)
        )
    else:
        raise ValueError("At least one transformation type must be True.")
    if n_dims % 1 != 0:
        raise ValueError("Could not infer dimensionality")
    return int(n_dims)


@script
def embed_transformation(matrix: Tensor, target_shape: List[int]) -> Tensor:
    """Embed transformation into larger dimensional space

    Args:
        matrix: Tensor with shape ([batch_size, ]n_dims, n_dims, ...)
        target_shape: Target matrix shape

    Returns: Tensor with shape (batch_size, *target_shape)
    """
    if len(target_shape) != 2:
        raise ValueError("Matrix shape must be two dimensional.")
    matrix = move_channels_last(matrix, 2)
    matrix, batch_dimensions_shape = merge_batch_dimensions(matrix, 2)
    batch_size = matrix.size(0)
    n_rows_needed = target_shape[0] - matrix.size(1)
    n_cols_needed = target_shape[1] - matrix.size(2)
    if n_rows_needed == 0 and n_cols_needed == 0:
        return matrix
    rows = cat(
        [
            zeros(
                n_rows_needed,
                min(matrix.size(2), matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                n_rows_needed,
                max(0, matrix.size(2) - matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=1,
    ).expand(batch_size, -1, -1)
    cols = cat(
        [
            zeros(
                min(target_shape[0], matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                max(0, target_shape[0] - matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=0,
    ).expand(batch_size, -1, -1)
    embedded_matrix = cat([cat([matrix, rows], dim=1), cols], dim=2)
    embedded_matrix = unmerge_batch_dimensions(
        embedded_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return move_channels_first(embedded_matrix, 2)


@script
def convert_to_homogenous_coordinates(coordinates: Tensor) -> Tensor:
    """Converts the coordinates to homogenous coordinates

    Args:
        coordinates: Tensor with shape (batch_size, n_channels, *)
        channels_first: Whether to have channels first, default True

    Returns: Tensor with shape (batch_size, n_channels + 1, *)
    """
    coordinates = move_channels_last(coordinates)
    coordinates, batch_dimensions_shape = merge_batch_dimensions(coordinates)
    homogenous_coordinates = cat(
        [
            coordinates,
            ones(1, device=coordinates.device, dtype=coordinates.dtype).expand(
                coordinates.size(0), 1
            ),
        ],
        dim=-1,
    )
    homogenous_coordinates = unmerge_batch_dimensions(
        homogenous_coordinates, batch_dimensions_shape=batch_dimensions_shape
    )
    return move_channels_first(homogenous_coordinates)


@script
def generate_translation_matrix(translations: Tensor) -> Tensor:
    """Generator homogenous translation matrix with given translations

    Args:
        translations: Tensor with shape (batch_size, n_dims, ...)

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    translations = move_channels_last(translations)
    translations, batch_dimensions_shape = merge_batch_dimensions(translations)
    batch_size = translations.size(0)
    n_dims = translations.size(1)
    homogenous_translation = convert_to_homogenous_coordinates(coordinates=translations)
    translation_matrix = cat(
        [
            cat(
                [
                    eye(n_dims, device=translations.device, dtype=translations.dtype),
                    zeros(1, n_dims, device=translations.device, dtype=translations.dtype),
                ],
                dim=0,
            ).expand(batch_size, -1, -1),
            homogenous_translation[..., None],
        ],
        dim=2,
    ).view(-1, n_dims + 1, n_dims + 1)
    translation_matrix = unmerge_batch_dimensions(
        translation_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return move_channels_first(translation_matrix, 2)


@script
def generate_rotation_matrix(rotations: Tensor) -> Tensor:
    """Generator rotation matrix from given rotations

    Args:
        rotations: Tensor with shape (batch_size, n_rotation_axes, ...)

    Returns: Tensor with shape (batch_size, n_dims, n_dims, ...)
    """
    rotations = move_channels_last(rotations)
    rotations, batch_dimensions_shape = merge_batch_dimensions(rotations)
    batch_size = rotations.size(0)
    n_dims = calculate_n_dims(rotations.size(1), AffineTransformationTypeDefinition.only_rotation())
    non_diagonal_indices = cat(
        (triu_indices(n_dims, n_dims, 1), tril_indices(n_dims, n_dims, -1)), dim=1
    )
    log_rotation_matrix = zeros(
        batch_size, n_dims, n_dims, device=rotations.device, dtype=rotations.dtype
    )
    log_rotation_matrix[:, non_diagonal_indices[0], non_diagonal_indices[1]] = cat(
        (rotations, -rotations), dim=1
    )
    rotation_matrix = torch_matrix_exp(log_rotation_matrix)
    rotation_matrix = unmerge_batch_dimensions(
        rotation_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return move_channels_first(rotation_matrix, 2)


@script
def generate_scale_and_shear_matrix(scales_and_shears: Tensor) -> Tensor:
    """Generator scale matrix from given scales and shears

    Args:
        scales_and_shears: Tensor with shape (batch_size, n_scale_and_shear_axes, ...)

    Returns: Tensor with shape (batch_size, n_dims, n_dims, ...)
    """
    scales_and_shears = move_channels_last(scales_and_shears)
    scales_and_shears, batch_dimensions_shape = merge_batch_dimensions(scales_and_shears)
    n_dims = calculate_n_dims(
        scales_and_shears.size(1),
        AffineTransformationTypeDefinition(
            translation=False, rotation=False, scale=True, shear=True
        ),
    )
    non_diagonal_indices = cat(
        (triu_indices(n_dims, n_dims, 1), tril_indices(n_dims, n_dims, -1)), dim=1
    )
    diagonal = scales_and_shears[:, :n_dims]
    off_diagonal = scales_and_shears[:, n_dims:]
    log_scale_and_shear_matrix = diag_embed(diagonal)
    log_scale_and_shear_matrix[:, non_diagonal_indices[0], non_diagonal_indices[1]] = cat(
        (off_diagonal, off_diagonal), dim=1
    )
    scale_and_shear_matrix = torch_matrix_exp(log_scale_and_shear_matrix)
    scale_and_shear_matrix = unmerge_batch_dimensions(
        scale_and_shear_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return move_channels_first(scale_and_shear_matrix, 2)


@script
def generate_scale_matrix(scales: Tensor) -> Tensor:
    """Generator scale matrix from given scales

    Args:
        scales: Tensor with shape (batch_size, n_scale_and_shear_axes, ...)

    Returns: Tensor with shape (batch_size, n_dims, n_dims, ...)
    """
    scales = move_channels_last(scales)
    scale_matrix = diag_embed(scales)
    return move_channels_first(scale_matrix, num_channel_dims=2)


@script
def _update_transformation(
    transformation: Optional[Tensor], new_transformation: Tensor
) -> Optional[Tensor]:
    if transformation is not None:
        if transformation.shape != new_transformation.shape:
            transformation = embed_transformation(transformation, new_transformation.shape[1:])
        transformation = matmul(transformation, new_transformation)  # type: ignore
    else:
        transformation = new_transformation
    return transformation


@script
def generate_affine_transformation_matrix(
    parameters: Tensor, transformation_type: AffineTransformationTypeDefinition
) -> Tensor:
    """Generates affine transformation matrix from correspoding
    euclidean space

    When translation, rotation, and shear are all True:
    For n_dims == 2, n_params = 2 + 1 + 3 = 6
    For n_dims == 3, n_params = 3 + 3 + 6 = 12

    Args:
        parameters: Tensor with shape (batch_size, n_params, ...)
        transformation_type: Type of affine transformation matrix to generate

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    parameters = move_channels_last(parameters)
    parameters, batch_dimensions_shape = merge_batch_dimensions(parameters)
    n_dims = calculate_n_dims(parameters.size(1), transformation_type)
    transformation: Optional[Tensor] = None
    n_parameters_used = 0
    if transformation_type.shear:
        if not transformation_type.scale:
            raise NotImplementedError(
                "Used parametrization method does not allow generating only shear without scaling"
            )
        n_scale_and_shear_params = calculate_n_parameters(
            n_dims,
            AffineTransformationTypeDefinition(
                translation=False, rotation=False, scale=True, shear=True
            ),
        )
        transformation = _update_transformation(
            transformation,
            generate_scale_and_shear_matrix(parameters[:, :n_scale_and_shear_params]),
        )
        n_parameters_used += n_scale_and_shear_params
    elif transformation_type.scale:
        n_scale_params = calculate_n_parameters(
            n_dims, AffineTransformationTypeDefinition.only_scale()
        )
        transformation = _update_transformation(
            transformation, generate_scale_matrix(parameters[:, :n_scale_params].exp())
        )
        n_parameters_used += n_scale_params
    if transformation_type.rotation:
        n_rotation_params = calculate_n_parameters(
            n_dims, AffineTransformationTypeDefinition.only_rotation()
        )
        transformation = _update_transformation(
            transformation,
            generate_rotation_matrix(
                parameters[:, n_parameters_used : n_parameters_used + n_rotation_params]
            ),
        )
        n_parameters_used += n_rotation_params
    if transformation_type.translation:
        transformation = _update_transformation(
            transformation, generate_translation_matrix(parameters[:, n_parameters_used:])
        )
    if transformation is None:
        raise RuntimeError("Emtpy transformation is not allowed")
    affine_transformation_matrix = embed_transformation(transformation, (n_dims + 1, n_dims + 1))
    affine_transformation_matrix = unmerge_batch_dimensions(
        affine_transformation_matrix,
        batch_dimensions_shape=batch_dimensions_shape,
        num_channel_dims=2,
    )
    return move_channels_first(affine_transformation_matrix, 2)


@script
def compose_affine_transformation_matrices(
    transformation_1: Tensor, transformation_2: Tensor
) -> Tensor:
    """Compose two transformation matrices

    Args:
        transformation_1: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)
        transformation_2: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)

    Returns: transformation_1: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)
    """
    transformation_1 = move_channels_last(transformation_1, 2)
    transformation_2 = move_channels_last(transformation_2, 2)
    composed = matmul(transformation_1, transformation_2)
    return move_channels_first(composed, 2)
