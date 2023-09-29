"""Tests for cubic b-spline upper bound computation algorithm"""

from itertools import product
from typing import Sequence
from unittest import TestCase

from torch import cat, empty, float64, tensor

from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory
from algorithm.composable_mapping.finite_difference import (
    estimate_spatial_jacobian_matrices_for_mapping,
)
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.cubic_b_spline_control_point_upper_bound import compute_max_control_point_value
from algorithm.cubic_spline_upsampling import CubicSplineUpsampling
from algorithm.interpolator import EmptyInterpolator
from tests.test_categories import lengthy


class CubicSplineCoefficientsTests(TestCase):
    """Tests for cubic spline coefficients"""

    def _compute_true_2d_upper_bound(self, upsampling_factors: Sequence[int]) -> float:
        max_lipschitz_constant = 0.0
        upsampling_factors_tensor = tensor(upsampling_factors, dtype=float64)
        for flattened_volume in product((-1, 1), repeat=16):
            volume = tensor(flattened_volume, dtype=float64).view(1, 1, 4, 4)
            upsampling = CubicSplineUpsampling(upsampling_factor=upsampling_factors, dtype=float64)
            upsampling.to(float64)

            upsampled_volume = upsampling(volume, apply_prefiltering=False) * 2 / 4
            stacked_upsampled_volume = cat([upsampled_volume] * 2, dim=1)
            coordinate_system = CoordinateSystemFactory.centered_normalized(
                original_grid_shape=upsampled_volume.shape[2:],
                voxel_size=1 / (2 * upsampling_factors_tensor),
                dtype=float64,
            )
            mapping = ComposableFactory.create_volume(
                data=stacked_upsampled_volume,
                coordinate_system=coordinate_system,
                grid_mapping_args=GridMappingArgs(
                    interpolator=EmptyInterpolator(),
                    mask_outside_fov=False,
                ),
            )
            batch_size = volume.size(0)
            n_dims = volume.ndim - 2
            jacobians = empty(
                (batch_size, 2, n_dims, 2**n_dims)
                + tuple(dim_size - 1 for dim_size in upsampled_volume.shape[2:]),
                dtype=volume.dtype,
            )
            estimate_spatial_jacobian_matrices_for_mapping(
                mapping=mapping,
                coordinate_system=coordinate_system,
                other_dims="crop_both",
                out=jacobians,
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
            compute_max_control_point_value(upsampling_factors=(7, 17), dtype=float64).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(16, 16)),
            compute_max_control_point_value(upsampling_factors=(16, 16), dtype=float64).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(1, 1)),
            compute_max_control_point_value(upsampling_factors=(1, 1), dtype=float64).item(),
            places=10,
        )
        self.assertAlmostEqual(
            self._compute_true_2d_upper_bound(upsampling_factors=(220, 256)),
            compute_max_control_point_value(upsampling_factors=(220, 256), dtype=float64).item(),
            places=10,
        )
