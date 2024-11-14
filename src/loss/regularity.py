"""Rigidity losses"""

from typing import Any, Mapping, Optional

from composable_mapping import (
    DataFormat,
    GridComposableMapping,
    LimitDirection,
    LinearInterpolator,
    OriginalShape,
    estimate_coordinate_mapping_spatial_derivatives,
)
from torch import Tensor, zeros

from algorithm.ndv import (
    calculate_jacobian_determinants,
    calculate_non_diffeomorphic_volume,
)
from loss.interface import IRegularityLoss
from util.dimension_order import get_other_than_batch_dim
from util.optional import optional_add

from .util import build_default_params, handle_params


class NDVLoss(IRegularityLoss):
    """Penalizes NDV of a deformation field"""

    def __init__(
        self,
        ndv_threshold: float = 0.0,
    ) -> None:
        self._ndv_threshold = ndv_threshold

    def __call__(
        self,
        mapping: GridComposableMapping,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        jacobians = calculate_jacobian_determinants(
            mapping.sample(DataFormat.voxel_displacements()).generate_values()
        )
        ndv = calculate_non_diffeomorphic_volume(
            jacobians,
            threshold=self._ndv_threshold,
        )
        return ndv.mean()


class _BaseGradientLoss(IRegularityLoss):
    """Gradient penalty on displacement field

    Args:
        regularize_flow: If True, assumes that the input mappings are not true
            coordinate mappings but instead mappings into displacements or flows.
    """

    def __init__(self, regularize_flow: bool) -> None:
        self._regularize_flow = regularize_flow
        self._default_params = build_default_params(
            none_ignored_params={"weight": 1.0},
        )

    def __call__(
        self,
        mapping: GridComposableMapping,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Calculate gradient loss

        Args:
            mapping: Mapping for which to calculate the gradient loss
            coordinate_system: Defines sampling locations for calculating
                the loss
        """
        params = handle_params(params, self._default_params)
        n_dims = len(mapping.coordinate_system.spatial_shape)
        loss: Optional[Tensor] = None
        for spatial_dim in range(n_dims):
            dim_gradients = estimate_coordinate_mapping_spatial_derivatives(
                mapping,
                spatial_dim=spatial_dim,
                target=mapping.coordinate_system.reformat(
                    spatial_shape=[
                        OriginalShape() - 1 if dim == spatial_dim else OriginalShape()
                        for dim in range(n_dims)
                    ]
                ),
                limit_direction=LimitDirection.left(),
                sampler=LinearInterpolator(mask_extrapolated_regions_for_empty_volume_mask=False),
            ).generate_values()
            if not self._regularize_flow:
                constant_dim_substraction = zeros(
                    n_dims, dtype=dim_gradients.dtype, device=dim_gradients.device
                )
                constant_dim_substraction[spatial_dim] = 1.0
                constant_dim_substraction = constant_dim_substraction.view(-1, *(1,) * n_dims)
                dim_gradients = dim_gradients - constant_dim_substraction
            loss = optional_add(
                loss, dim_gradients.square().mean(dim=get_other_than_batch_dim(dim_gradients))
            )
        assert loss is not None
        return (params["weight"] * loss / n_dims).mean()


class GradientDeformationLoss(_BaseGradientLoss):
    """Gradient penalty on deformations"""

    def __init__(self) -> None:
        super().__init__(regularize_flow=False)


class GradientFlowLoss(_BaseGradientLoss):
    """Gradient penalty on flows"""

    def __init__(self) -> None:
        super().__init__(regularize_flow=True)
