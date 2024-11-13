"""Implementations for sampling affine transformations"""

from math import pi
from typing import NamedTuple, Sequence

from torch import Generator, Tensor, cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import randn, tensor

from algorithm.affine_transformation import (
    compose_affine_transformation_matrices,
    embed_transformation,
    generate_rotation_matrix,
    generate_scale_and_shear_matrix,
    generate_translation_matrix,
)


class AffineTransformationSamplingArguments(NamedTuple):
    """Sampling arguments for sampling affine transformations

    Arguments:
        translation_stds: Translation standard deviations along each dimension
        rotation_stds: Rotation standard deviations along each dimension in degrees
        logscale_stds: Logscale standard deviations along each dimension
        logshear_stds: Logshear standard deviations along each dimension
    """

    translation_stds: Sequence[float]
    rotation_stds: Sequence[float]
    logscale_stds: Sequence[float]
    logshear_stds: Sequence[float]


def sample_random_affine_transformation(
    n_transformations: int,
    arguments: AffineTransformationSamplingArguments,
    generator: Generator | None = None,
    dtype: torch_dtype | None = None,
    device: torch_device | None = None,
) -> Tensor:
    """Sample random elastic deformation

    Samples a random deformation by first generating downsampled white noise
    which is then upsampled using cubic spline upsamling and further smoothed
    using gaussian smoothing. Standard deviations are given in world coordinates.
    """
    unnormalized_translations = randn(
        size=(n_transformations, len(arguments.rotation_stds)),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    dtype = unnormalized_translations.dtype
    device = unnormalized_translations.device

    def _to_tensor(float_sequence: Sequence[float]) -> Tensor:
        return tensor(
            float_sequence,
            dtype=dtype,
            device=device,
        )

    translations = unnormalized_translations * _to_tensor(arguments.translation_stds)
    rotations = (
        randn(
            size=(n_transformations, len(arguments.rotation_stds)),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        * _to_tensor(arguments.rotation_stds)
        * pi
        / 180
    )
    logscales = randn(
        size=(n_transformations, len(arguments.logscale_stds)),
        generator=generator,
        dtype=dtype,
        device=device,
    ) * _to_tensor(arguments.logscale_stds)
    logshears = randn(
        size=(n_transformations, len(arguments.logshear_stds)),
        generator=generator,
        dtype=dtype,
        device=device,
    ) * _to_tensor(arguments.logshear_stds)
    scale_and_shear_matrix = generate_scale_and_shear_matrix(
        cat((logscales, logshears), dim=1)
    )
    rotation_matrix = generate_rotation_matrix(rotations)
    translation_matrix = generate_translation_matrix(translations)
    transformation_matrix = compose_affine_transformation_matrices(
        embed_transformation(
            compose_affine_transformation_matrices(
                scale_and_shear_matrix, rotation_matrix
            ),
            target_shape=translation_matrix.shape[1:],
        ),
        translation_matrix,
    )
    return transformation_matrix
