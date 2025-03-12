"""Tests for cubic b-spline upper bound computation algorithm"""

from itertools import product
from typing import Sequence
from unittest import TestCase

from composable_mapping import (
    CoordinateSystem,
    CubicSplineSampler,
    DataFormat,
    End,
    sampling_cache,
    LimitDirection,
    LinearInterpolator,
    OriginalShape,
    SamplableVolume,
    Start,
    estimate_spatial_jacobian_matrices,
    samplable_volume,
)
from torch import float64, stack, tensor
from tqdm import tqdm  # type: ignore

from algorithm.cubic_b_spline_control_point_upper_bound import (
    compute_max_control_point_value,
)
from tests.test_categories import lengthy


class CubicSplineCoefficientsTests(TestCase):
    """Tests for cubic spline coefficients"""

    def _compute_true_2d_upper_bound(self, upsampling_factors: Sequence[int]) -> float:
        max_lipschitz_constant = 0.0
        for flattened_volume in tqdm(list(product((-1, 1), repeat=16))):
            with sampling_cache():
                volume = (
                    tensor(flattened_volume, dtype=float64)
                    .view(1, 1, 4, 4)
                    .expand(1, 2, 4, 4)
                )
                coordinate_system = CoordinateSystem.centered_normalized(
                    volume.shape[2:],
                    voxel_size=1.0,
                    dtype=float64,
                    device=volume.device,
                )
                upsampled_coordinate_system = coordinate_system.reformat(
                    upsampling_factor=upsampling_factors,
                )
                mapping = samplable_volume(
                    volume,
                    coordinate_system=coordinate_system,
                    sampler=CubicSplineSampler(prefilter=False),
                    data_format=DataFormat.voxel_displacements(),
                ).resample_to(
                    upsampled_coordinate_system,
                    sampler=LinearInterpolator(mask_extrapolated_regions=False),
                )
                mapping = SamplableVolume(
                    mapping.sample(DataFormat.world_displacements()),
                    coordinate_system=upsampled_coordinate_system,
                    sampler=LinearInterpolator(mask_extrapolated_regions=False),
                )
                jacobians_corner_left_left = estimate_spatial_jacobian_matrices(
                    mapping,
                    target=upsampled_coordinate_system.reformat(
                        spatial_shape=OriginalShape() - 1, reference=(Start(), Start())
                    ),
                    sampler=LinearInterpolator(
                        mask_extrapolated_regions=False,
                        limit_direction=lambda spatial_dim: [
                            LimitDirection.right(),
                            LimitDirection.right(),
                        ][spatial_dim],
                    ),
                ).generate_values()
                jacobians_corner_left_right = estimate_spatial_jacobian_matrices(
                    mapping,
                    target=upsampled_coordinate_system.reformat(
                        spatial_shape=OriginalShape() - 1, reference=(Start(), End())
                    ),
                    sampler=LinearInterpolator(
                        mask_extrapolated_regions=False,
                        limit_direction=lambda spatial_dim: [
                            LimitDirection.right(),
                            LimitDirection.left(),
                        ][spatial_dim],
                    ),
                ).generate_values()
                jacobians_corner_right_left = estimate_spatial_jacobian_matrices(
                    mapping,
                    target=upsampled_coordinate_system.reformat(
                        spatial_shape=OriginalShape() - 1, reference=(End(), Start())
                    ),
                    sampler=LinearInterpolator(
                        mask_extrapolated_regions=False,
                        limit_direction=lambda spatial_dim: [
                            LimitDirection.left(),
                            LimitDirection.right(),
                        ][spatial_dim],
                    ),
                ).generate_values()
                jacobians_corner_right_right = estimate_spatial_jacobian_matrices(
                    mapping,
                    target=upsampled_coordinate_system.reformat(
                        spatial_shape=OriginalShape() - 1, reference=(End(), End())
                    ),
                    sampler=LinearInterpolator(
                        mask_extrapolated_regions=False,
                        limit_direction=lambda spatial_dim: [
                            LimitDirection.left(),
                            LimitDirection.left(),
                        ][spatial_dim],
                    ),
                ).generate_values()
                jacobians = stack(
                    [
                        jacobians_corner_left_left,
                        jacobians_corner_left_right,
                        jacobians_corner_right_left,
                        jacobians_corner_right_right,
                    ],
                    dim=3,
                )
                lipschitz_constant = jacobians.abs().sum(dim=2).max().item()
                if lipschitz_constant > max_lipschitz_constant:
                    max_lipschitz_constant = lipschitz_constant
        return 1 / max_lipschitz_constant

    @lengthy
    def test_cubic_spline_coefficients_grad(self) -> None:
        """Test that upper bound is correct for different upsampling factors"""
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(7, 17)),
            compute_max_control_point_value(
                upsampling_factors=(7, 17), dtype=float64
            ).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(16, 16)),
            compute_max_control_point_value(
                upsampling_factors=(16, 16), dtype=float64
            ).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(1, 1)),
            compute_max_control_point_value(
                upsampling_factors=(1, 1), dtype=float64
            ).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(220, 256)),
            compute_max_control_point_value(
                upsampling_factors=(220, 256), dtype=float64
            ).item(),
            places=10,
        )
