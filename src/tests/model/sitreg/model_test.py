"""Tests for SITReg model"""

from unittest import TestCase

from composable_mapping import GridComposableMapping
from deformation_inversion_layer.fixed_point_iteration import AndersonSolver
from torch import rand
from torch.testing import assert_close

from algorithm.affine_transformation import AffineTransformationTypeDefinition
from model.activation import ReLUFactory
from model.normalizer import GroupNormalizerFactory
from model.sitreg import SITReg
from model.sitreg.feature_extractor import EncoderFeatureExtractor


class SITRegTests(TestCase):
    """Tests for SITReg model"""

    def test_inverse_consistency(self) -> None:
        """Test that swapping the input arguments swaps the outputs"""
        image_1 = rand(16, 3, 64, 64)
        image_2 = rand(16, 3, 64, 64)
        feature_extractor = EncoderFeatureExtractor(
            n_input_channels=3,
            activation_factory=ReLUFactory(),
            n_features_per_resolution=[8, 16, 32],
            n_convolutions_per_resolution=[2, 2, 2],
            input_shape=[64, 64],
            normalizer_factory=GroupNormalizerFactory(2),
        )
        symmetric_network = SITReg(
            feature_extractor=feature_extractor,
            n_transformation_convolutions_per_resolution=[2, 2, 2],
            n_transformation_features_per_resolution=[16, 32, 64],
            max_control_point_multiplier=0.99,
            affine_transformation_type=AffineTransformationTypeDefinition.full(),
            input_voxel_size=(1.0, 2.0),
            input_shape=(64, 64),
            transformation_downsampling_factor=(1.0, 1.0),
            forward_fixed_point_solver=AndersonSolver(),
            backward_fixed_point_solver=AndersonSolver(),
            activation_factory=ReLUFactory(),
            normalizer_factory=GroupNormalizerFactory(2),
        )
        image_coordinate_system = symmetric_network.image_coordinate_system
        forward_mapping: GridComposableMapping
        inverse_mapping: GridComposableMapping
        reverse_forward_mapping: GridComposableMapping
        reverse_inverse_mapping: GridComposableMapping
        ((forward_mapping, inverse_mapping),) = symmetric_network(image_1=image_1, image_2=image_2)
        ((reverse_forward_mapping, reverse_inverse_mapping),) = symmetric_network(
            image_1=image_2,
            image_2=image_1,
        )
        assert_close(
            forward_mapping(image_coordinate_system.grid).generate_values(),
            reverse_inverse_mapping(image_coordinate_system.grid).generate_values(),
        )
        assert_close(
            inverse_mapping(image_coordinate_system.grid).generate_values(),
            reverse_forward_mapping(image_coordinate_system.grid).generate_values(),
        )
