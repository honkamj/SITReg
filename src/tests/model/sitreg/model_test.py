"""Tests for SITReg model"""


from unittest import TestCase

from torch import rand
from torch.nn.functional import relu
from torch.testing import assert_close

from algorithm.affine_transformation import AffineTransformationTypeDefinition
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.composable_mapping.interface import IComposableMapping
from algorithm.fixed_point_solver import AndersonSolver
from algorithm.interpolator import LinearInterpolator
from model.sitreg import SITReg
from model.sitreg.feature_extractor import EncoderFeatureExtractor
from model.normalizer import GroupNormalizerFactory


class SITRegTests(TestCase):
    """Tests for SITReg model"""

    def test_inverse_consistency(self) -> None:
        """Test that swapping the input arguments swaps the outputs"""
        image_1 = rand(16, 3, 64, 64)
        image_2 = rand(16, 3, 64, 64)
        feature_extractor = EncoderFeatureExtractor(
            n_input_channels=3,
            activation=relu,
            n_features_per_resolution=[8, 16, 32],
            n_convolutions_per_resolution=[2, 2, 2],
            input_shape=[64, 64],
            normalizer_factory=GroupNormalizerFactory(2),
        )
        symmetric_network = SITReg(
            feature_extractor=feature_extractor,
            n_transformation_convolutions_per_resolution=[2, 2, 2],
            n_transformation_features_per_resolution=[16, 32, 64],
            max_displacements=[6.0, 6.0, 6.0],
            affine_transformation_type=AffineTransformationTypeDefinition.full(),
            input_voxel_size=(1.0, 2.0),
            input_shape=(64, 64),
            transformation_downsampling_factor=(1.0, 1.0),
            transformation_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(padding_mode="border"), mask_outside_fov=False
            ),
            volume_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(padding_mode="border"), mask_outside_fov=False
            ),
            forward_fixed_point_solver=AndersonSolver(),
            backward_fixed_point_solver=AndersonSolver(),
            activation=relu,
        )
        image_coordinate_system = symmetric_network.image_coordinate_system
        forward_mapping: IComposableMapping
        inverse_mapping: IComposableMapping
        reverse_forward_mapping: IComposableMapping
        reverse_inverse_mapping: IComposableMapping
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
