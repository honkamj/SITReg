"""Tests for rigidity penalty"""

from unittest import TestCase

from torch import device as torch_device
from torch import rand, tensor
from torch.testing import assert_close

from algorithm.affine_transformation import AffineTransformationTypeDefinition
from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.interpolator import LinearInterpolator
from loss.regularity import GradientDeformationLoss, JacobianLoss


class JacobianLossTests(TestCase):
    """Tests for Jacobian based losses"""

    def test_affinity_penalty_zero_for_affine(self) -> None:
        """Test that affinity penalty is zero for affine transformations"""
        affine = ComposableFactory.create_affine_from_parameters(
            parameters=(2 * rand(1, 6) - 1),
            transformation_type=AffineTransformationTypeDefinition.full(),
        )
        test_coordinate_system = CoordinateSystemFactory.centered_normalized([64, 128], [1.2, 3.4])
        for central, other_dims in ((False, "average"), (True, "crop")):
            loss = JacobianLoss(
                orthonormality_weight=None,
                properness_weight=None,
                affinity_weight=1.0,
                invertibility_weight=None,
                central=central,
                other_dims=other_dims,
            )(affine, test_coordinate_system, device=torch_device("cpu"))
            assert_close(loss, tensor(0.0, dtype=loss.dtype, device=loss.device))

    def test_penalty_zero_for_rotation(self) -> None:
        """Test that affinity penalty is zero for affine transformations"""
        affine = ComposableFactory.create_affine_from_parameters(
            parameters=(2 * rand(1, 3) - 1),
            transformation_type=AffineTransformationTypeDefinition.only_rotation(),
        )
        test_coordinate_system = CoordinateSystemFactory.centered_normalized(
            [10, 12, 14], [1.2, 3.4, 0.5]
        )
        for central, other_dims in ((False, "average"), (True, "crop")):
            loss = JacobianLoss(
                orthonormality_weight=1.0,
                properness_weight=1.0,
                affinity_weight=1.0,
                invertibility_weight=None,
                central=central,
                other_dims=other_dims,
            )(affine, test_coordinate_system, device=torch_device("cpu"))
            assert_close(loss, tensor(0.0, dtype=loss.dtype, device=loss.device))

    def test_penalty_non_zero_for_random_deformation(self) -> None:
        """Test that affinity penalty is non-zero for non-affine transformations"""
        test_coordinate_system = CoordinateSystemFactory.centered_normalized(
            [10, 12, 14], [1.2, 3.4, 0.5]
        )
        random_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=(2 * rand(1, 3, 10, 12, 14) - 1) * 0.5,
            coordinate_system=test_coordinate_system,
            grid_mapping_args=GridMappingArgs(interpolator=LinearInterpolator()),
        )
        for central, other_dims in ((False, "average"), (True, "crop")):
            loss = JacobianLoss(
                orthonormality_weight=0.01,
                properness_weight=0.1,
                affinity_weight=1.0,
                invertibility_weight=None,
                central=central,
                other_dims=other_dims,
            )(random_mapping, test_coordinate_system, device=torch_device("cpu"))
            self.assertGreater(float(loss), 10)


class GradientLossTests(TestCase):
    """Tests for gradient based losses"""

    def test_gradient_penalty_non_zero_for_random_deformation(self) -> None:
        """Test that penalty is non-zero for random transformations"""
        test_coordinate_system = CoordinateSystemFactory.centered_normalized(
            [10, 12, 14], [1.2, 3.4, 0.5]
        )
        random_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=(2 * rand(1, 3, 10, 12, 14) - 1) * 0.5,
            coordinate_system=test_coordinate_system,
            grid_mapping_args=GridMappingArgs(interpolator=LinearInterpolator()),
        )
        for central in (False, True):
            loss = GradientDeformationLoss(central=central)(
                random_mapping, test_coordinate_system, device=torch_device("cpu")
            )
            self.assertGreater(float(loss), 0.04)
