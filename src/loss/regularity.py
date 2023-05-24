"""Rigidity losses"""

from itertools import combinations_with_replacement
from typing import Optional, Sequence

from torch import Tensor, relu, zeros
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import eye, matmul, mean, square

from algorithm.composable_mapping.finite_difference import (
    estimate_spatial_derivatives_for_mapping,
    estimate_spatial_jacobian_matrices_for_mapping,
)
from algorithm.composable_mapping.interface import IComposableMapping, VoxelCoordinateSystem
from algorithm.everywhere_differentiable_determinant import calculate_determinant
from algorithm.finite_difference import estimate_spatial_derivatives
from loss.interface import IRegularityLoss
from util.dimension_order import (
    get_other_than_batch_dim,
    index_by_channel_dims,
    move_channels_last,
    num_spatial_dims,
)
from util.optional import optional_add


def affinity(
    jacobian_matrices: Tensor,
    spacing: Sequence[float],
    other_dims: str = "average",
    central: bool = True,
) -> Tensor:
    """Affinity (bending energy) penalty for deformation fields penalizing non-affine deformations

    Args:
        jacobian_matrices: Local Jacobian matrices of the deformation mapping,
                Tensor with shape (batch_size, n_dims, n_dims, *any_shape)
        spacing: Jacobian matrix grid spacing
        other_dims: See option other_dims of algorithm.finite_difference
        central: See option central of algorithm.finite_difference

    Returns:
        Tensor with shape (batch_size,)
    """
    n_dims = num_spatial_dims(jacobian_matrices.ndim, 2)
    loss: Tensor | None = None
    channels_dim = index_by_channel_dims(jacobian_matrices.ndim, 0, 2)
    n_terms = n_dims**2
    for i, j in combinations_with_replacement(range(n_dims), 2):
        gradient_volume = estimate_spatial_derivatives(
            volume=jacobian_matrices.select(channels_dim + 1, i),
            spacing=float(spacing[j]),
            spatial_dim=j,
            n_channel_dims=1,
            other_dims=other_dims,
            central=central,
        )
        if i == j:
            loss = optional_add(
                loss,
                mean(square(gradient_volume), dim=get_other_than_batch_dim(gradient_volume))
                / n_terms,
            )
        else:
            loss = optional_add(
                loss,
                2
                * mean(square(gradient_volume), dim=get_other_than_batch_dim(gradient_volume))
                / n_terms,
            )
    assert loss is not None
    return loss


def properness(jacobian_determinants: Tensor) -> Tensor:
    """Properness penalty for deformation fields penalizing locally volume changing deformations

    Args:
        jacobian_determinants: Local Jacobian determinants of the deformation mapping,
                Tensor with shape (batch_size, 1, *any_shape)

    Returns:
        Tensor with size (batch_size,)
    """
    return mean(
        square(jacobian_determinants - 1), dim=get_other_than_batch_dim(jacobian_determinants)
    )


def invertibility(jacobian_determinants: Tensor) -> Tensor:
    """Invertibility penalty for deformation fields penalizing negative determinants

    Args:
        jacobian_determinants: Local Jacobian determinants of the deformation mapping,
                Tensor with shape (batch_size, 1, *any_shape)

    Returns:
        Tensor with size (batch_size,)
    """
    return mean(relu(-jacobian_determinants), dim=get_other_than_batch_dim(jacobian_determinants))


def orthonormality(jacobian_matrices: Tensor) -> Tensor:
    """Orthonormality penalty for deformation field penalizing locally non-orthonormal deformations

    Args:
        jacobian_matrices: Local Jacobian matrices of the deformation mapping,
                Tensor with shape (batch_size, n_dims, n_dims, *any_shape)

    Returns:
        Tensor with size (batch_size,)
    """
    jacobian_matrices = move_channels_last(jacobian_matrices, 2)
    n_dims = jacobian_matrices.size(-1)
    identity_matrix = eye(n_dims, device=jacobian_matrices.device)
    orthonormality_product = (
        matmul(jacobian_matrices, jacobian_matrices.transpose(-1, -2)) - identity_matrix
    )
    return mean(orthonormality_product, dim=get_other_than_batch_dim(orthonormality_product))


class JacobianLoss(IRegularityLoss):
    """Regularizatoin losses which are based on Jacobian matrix

    The orthonormality, properness and affinity terms are from the paper:

    Staring, Marius, Stefan Klein, and Josien PW Pluim. "A rigidity penalty term
    for nonrigid registration." (2007)

    Invertibility term is the "local orientation consistency" loss from

    Mok, Tony CW, and Albert Chung. "Fast symmetric diffeomorphic image
    registration with convolutional neural networks." (2020)

    Arguments:
        orthonormality_weight: Weight of the orthonormality term
        properness_weight: Weight of the properness term
        affinity_weight: Weight of the affinity term
        invertibility_weight: Weight of the invertibility term
        other_dims: See option other_dims of algorithm.finite_difference
        central: See option central of algorithm.finite_difference
    """

    def __init__(
        self,
        orthonormality_weight: Optional[float] = 1e-2,
        properness_weight: Optional[float] = 1e-1,
        affinity_weight: Optional[float] = 1.0,
        invertibility_weight: Optional[float] = None,
        other_dims: str = "average",
        central: bool = False,
    ) -> None:
        self._orthonormality_weight = orthonormality_weight
        self._properness_weight = properness_weight
        self._affinity_weight = affinity_weight
        self._invertibility_weight = invertibility_weight
        self._other_dims = other_dims
        self._central = central

    def __call__(
        self,
        mapping: IComposableMapping,
        coordinate_system: VoxelCoordinateSystem,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> Tensor:
        """Calculate rigidity loss

        Args:
            mapping: Mapping for which to calculate the rigidity
            coordinate_system: Defines sampling locations for calculating
                the loss
        """
        jacobian_matrices = estimate_spatial_jacobian_matrices_for_mapping(
            mapping=mapping,
            coordinate_system=coordinate_system,
            other_dims=self._other_dims,
            central=self._central,
            device=device,
            dtype=dtype,
        )
        loss: Tensor | None = None
        if self._orthonormality_weight is not None:
            loss = optional_add(
                loss, self._orthonormality_weight * orthonormality(jacobian_matrices).mean()
            )
        if self._affinity_weight is not None:
            loss = optional_add(
                loss,
                affinity(
                    jacobian_matrices,
                    spacing=coordinate_system.grid_spacing,
                    other_dims=self._other_dims,
                    central=self._central,
                ).mean(),
            )
        if self._properness_weight is not None or self._invertibility_weight is not None:
            jacobian_determinants = calculate_determinant(jacobian_matrices)
            if self._properness_weight is not None:
                loss = optional_add(
                    loss, self._properness_weight * properness(jacobian_determinants).mean()
                )
            if self._invertibility_weight is not None:
                loss = optional_add(
                    loss,
                    self._invertibility_weight * invertibility(jacobian_determinants).mean(),
                )
        assert loss is not None
        return loss


class _BaseGradientLoss(IRegularityLoss):
    """Gradient penalty on displacement field"""

    def __init__(self, regularize_flow: bool, central: bool) -> None:
        self._regularize_flow = regularize_flow
        self._central = central

    def __call__(
        self,
        mapping: IComposableMapping,
        coordinate_system: VoxelCoordinateSystem,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> Tensor:
        """Calculate gradient loss

        Args:
            mapping: Mapping for which to calculate the gradient loss
            coordinate_system: Defines sampling locations for calculating
                the loss
        """
        n_dims = len(coordinate_system.grid_spacing)
        loss: Optional[Tensor] = None
        for dim in range(n_dims):
            dim_gradients = estimate_spatial_derivatives_for_mapping(
                mapping=mapping,
                coordinate_system=coordinate_system,
                spatial_dim=dim,
                other_dims=None,
                central=self._central,
                device=device,
                dtype=dtype,
            )
            if not self._regularize_flow:
                constant_dim_substraction = zeros(
                    n_dims, dtype=dim_gradients.dtype, device=dim_gradients.device
                )
                constant_dim_substraction[dim] = 1.0
                constant_dim_substraction = constant_dim_substraction.view(-1, *(1,) * n_dims)
                dim_gradients = dim_gradients - constant_dim_substraction
            loss = optional_add(
                loss, dim_gradients.square().mean()
            )
        assert loss is not None
        return loss / n_dims


class GradientDeformationLoss(_BaseGradientLoss):
    """Gradient penalty on deformations"""

    def __init__(self, central: bool = False) -> None:
        super().__init__(regularize_flow=False, central=central)


class GradientFlowLoss(_BaseGradientLoss):
    """Gradient penalty on flows"""

    def __init__(self, central: bool = False) -> None:
        super().__init__(regularize_flow=True, central=central)
