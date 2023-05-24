"""Tests for finite difference derivatives"""

from unittest import TestCase

from torch import device as torch_device, eye
from torch.testing import assert_close
from util.dimension_order import broadcast_tensors_by_leading_dims
from algorithm.composable_mapping.finite_difference import (
    estimate_spatial_jacobian_matrices_for_mapping)
from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory

class FiniteDifferenceTests(TestCase):
    """Tests for finite difference derivate estimation"""

    def test_identity_jacobian_of_identity(self) -> None:
        """Test that deriving identity results in identity"""
        coordinate_system = CoordinateSystemFactory.centered_normalized(
            (10, 11, 12),
            (0.9, 1.0, 1.1)
        )
        matrices = estimate_spatial_jacobian_matrices_for_mapping(
            mapping=ComposableFactory.create_identity(),
            coordinate_system=coordinate_system,
            central=True,
            other_dims="crop",
            device=torch_device('cpu')
        )
        matrices, identity = broadcast_tensors_by_leading_dims(
            (matrices, eye(3)),
            num_leading_dims=2
        )
        assert_close(
            matrices,
            identity
        )
