"""Implementations for everywhere differentiable determinants of small matrices"""

from torch import Tensor

from util.dimension_order import channels_last, merged_batch_dimensions


def _calculate_determinant_1d(matrix: Tensor) -> Tensor:
    return matrix[:, 0]


def _calculate_determinant_2d(matrix: Tensor) -> Tensor:
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]


def _calculate_determinant_3d(matrix: Tensor) -> Tensor:
    return (
        matrix[:, 0, 0] * (matrix[:, 1, 1] * matrix[:, 2, 2] - matrix[:, 1, 2] * matrix[:, 2, 1])
        + matrix[:, 0, 1] * (matrix[:, 1, 2] * matrix[:, 2, 0] - matrix[:, 1, 0] * matrix[:, 2, 2])
        + matrix[:, 0, 2] * (matrix[:, 1, 0] * matrix[:, 2, 1] - matrix[:, 1, 1] * matrix[:, 2, 0])
    )


@channels_last(2, 1)
@merged_batch_dimensions(2, 1)
def calculate_determinant(matrix: Tensor) -> Tensor:
    """Calculate determinant of a matrix

    Args:
        matrix: Tensor with shape (batch_size, n_dims, n_dims, *any_shape)
        channels_first: Whether to have channels first, default True

    Returns:
        Tensor with shape (batch_size, 1, *any_shape)
    """
    n_dims = matrix.size(1)
    if n_dims == 1:
        return _calculate_determinant_1d(matrix)[..., None]
    if n_dims == 2:
        return _calculate_determinant_2d(matrix)[..., None]
    if n_dims == 3:
        return _calculate_determinant_3d(matrix)[..., None]
    raise NotImplementedError
