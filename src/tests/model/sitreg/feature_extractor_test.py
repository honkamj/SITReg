"""Tests for feature extractors"""

from unittest import TestCase

from torch import zeros

from algorithm.shape_logic import EncoderShapeLogic
from model.sitreg.feature_extractor import EncoderFeatureExtractor, UNetFeatureExtractor


class UnetFeatureExtractorTests(TestCase):
    """Tests for U-Net feature extractor"""

    def test_feature_shapes(self) -> None:
        """Test correct feature shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode="ceil", n_feature_levels=5, input_shape=input_shape, downsampling_factor=2
        )
        unet = UNetFeatureExtractor(
            n_input_channels=2,
            activation=lambda x: x,
            n_features_per_resolution=[2, 3, 4, 5, 6],
            n_convolutions_per_resolution=[2, 2, 2, 2, 2],
            input_shape=(15, 32, 17),
        )
        test_input = zeros(3, 2, 15, 32, 17)
        test_features = unet(test_input)
        for target_shape, test_feature in zip(shape_logic.calculate_shapes(), test_features):
            self.assertSequenceEqual(target_shape, test_feature.shape[2:])


class EncoderFeatureExtractorTests(TestCase):
    """Tests for Encoder feature extractor"""

    def test_ceil_feature_shapes(self) -> None:
        """Test correct feature shapes with ceil mode"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode="ceil", n_feature_levels=5, input_shape=input_shape, downsampling_factor=2
        )
        encoder = EncoderFeatureExtractor(
            n_input_channels=2,
            activation=lambda x: x,
            n_features_per_resolution=[2, 3, 4, 5, 6],
            n_convolutions_per_resolution=[2, 2, 2, 2, 2],
            input_shape=(15, 32, 17),
        )
        test_input = zeros(3, 2, 15, 32, 17)
        test_features = encoder(test_input)
        for target_shape, test_feature in zip(shape_logic.calculate_shapes(), test_features):
            self.assertSequenceEqual(target_shape, test_feature.shape[2:])
