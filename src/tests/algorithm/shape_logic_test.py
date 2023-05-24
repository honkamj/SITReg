"""Encoder tests"""

from unittest import TestCase

from torch import ones, zeros_like
from torch.nn.functional import (conv_transpose1d, conv_transpose2d,
                                 conv_transpose3d)
from torch.testing import assert_close

from algorithm.shape_logic import DecoderShapeLogic, EncoderShapeLogic


class EncoderShapeLogicTests(TestCase):
    """Tests for Encoder shape logic"""

    def test_ceil_shapes_logic(self) -> None:
        """Test correct ceil shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode='ceil',
            n_feature_levels=5,
            input_shape=input_shape,
            downsampling_factor=2)
        target_shapes = [
            (15, 32, 17),
            (8, 16, 9),
            (4, 8, 5),
            (2, 4, 3),
            (1, 2, 2)
        ]
        for shape, target_shape in zip(
                shape_logic.calculate_shapes(),
                target_shapes):
            self.assertSequenceEqual(
                shape,
                target_shape
            )


    def test_floor_shapes_logic(self) -> None:
        """Test correct floor shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode='floor',
            n_feature_levels=5,
            input_shape=input_shape,
            downsampling_factor=2)
        target_shapes = [
            (15, 32, 17),
            (7, 16, 8),
            (3, 8, 4),
            (1, 4, 2)
        ]
        for shape, target_shape in zip(
                shape_logic.calculate_shapes(),
                target_shapes):
            self.assertSequenceEqual(
                shape,
                target_shape
            )

    def test_ceil_pre_downsampling_paddings(self) -> None:
        """Test correct floor shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
                shape_mode='ceil',
                n_feature_levels=5,
                input_shape=input_shape,
                downsampling_factor=2)
        target_pads = [
            (1, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1)
        ]
        for earlier_downsamplings, target_pad in zip(range(4), target_pads):
            self.assertSequenceEqual(
                shape_logic.calculate_pre_downsampling_paddings(
                    earlier_downsamplings=earlier_downsamplings
                ),
                target_pad
            )

    def test_floor_pre_downsampling_paddings(self) -> None:
        """Ensure correct floor pre downsampling paddings"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode='floor',
            n_feature_levels=4,
            input_shape=input_shape,
            downsampling_factor=2)
        for earlier_downsamplings in range(3):
            self.assertSequenceEqual(
                shape_logic.calculate_pre_downsampling_paddings(
                    earlier_downsamplings=earlier_downsamplings
                ),
                (0, 0, 0)
            )


class DecoderShapeLogicTests(TestCase):
    """Tests for decoder shape logic"""
    def test_ceil_shapes_logic(self) -> None:
        """Test correct ceil shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode='ceil',
            n_feature_levels=5,
            input_shape=input_shape,
            downsampling_factor=2)
        target_shapes = [
            (15, 32, 17),
            (8, 16, 9),
            (4, 8, 5),
            (2, 4, 3),
            (1, 2, 2)
        ]
        for shape, target_shape in zip(
                shape_logic.calculate_shapes(),
                target_shapes):
            self.assertSequenceEqual(
                shape,
                target_shape
            )


    def test_floor_shapes_logic(self) -> None:
        """Test correct floor shapes"""
        input_shape = (15, 32, 17)
        shape_logic = EncoderShapeLogic(
            shape_mode='floor',
            n_feature_levels=5,
            input_shape=input_shape,
            downsampling_factor=2)
        target_shapes = [
            (15, 32, 17),
            (7, 16, 8),
            (3, 8, 4),
            (1, 4, 2)
        ]
        for shape, target_shape in zip(
                shape_logic.calculate_shapes(),
                target_shapes):
            self.assertSequenceEqual(
                shape,
                target_shape
            )
    def test_ceil_output_paddings(self) -> None:
        """Test correct ceil output paddings"""
        input_shape = (15, 32, 17)
        shape_logic = DecoderShapeLogic(
            EncoderShapeLogic(
                shape_mode='ceil',
                n_feature_levels=5,
                input_shape=input_shape,
                downsampling_factor=2))
        for upsamplings in range(4):
            self.assertSequenceEqual(
                shape_logic.upsampling_output_padding(
                    earlier_upsamplings=upsamplings
                ),
                (0, 0, 0)
            )

    def test_floor_output_paddings(self) -> None:
        """Test correct floor output paddings"""
        input_shape = (15, 32, 17)
        shape_logic = DecoderShapeLogic(
            EncoderShapeLogic(
                shape_mode='floor',
                n_feature_levels=4,
                input_shape=input_shape,
                downsampling_factor=2))
        target_pads = [
            (1, 0, 0),
            (1, 0, 0),
            (1, 0, 1)
        ]
        for upsamplings, target_pad in zip(range(4), target_pads):
            self.assertSequenceEqual(
                shape_logic.upsampling_output_padding(
                    earlier_upsamplings=upsamplings
                ),
                target_pad
            )

    def test_ceil_post_upsampling_crops(self) -> None:
        """Test correct crops for ceil"""
        input_shape = (15, 32, 17)
        shape_logic = DecoderShapeLogic(
            EncoderShapeLogic(
                shape_mode='ceil',
                n_feature_levels=5,
                input_shape=input_shape,
                downsampling_factor=2))
        target_crops = [
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 1)
        ]
        for upsamplings, target_crop in zip(range(1, 5), target_crops):
            self.assertSequenceEqual(
                shape_logic.calculate_post_upsampling_crops(
                    upsamplings=upsamplings
                ),
                target_crop
            )

    def test_floor_post_upsampling_crops(self) -> None:
        """Test correct crops for floor"""
        input_shape = (15, 32, 17)
        shape_logic = DecoderShapeLogic(
            EncoderShapeLogic(
                shape_mode='floor',
                n_feature_levels=4,
                input_shape=input_shape,
                downsampling_factor=2))
        for upsamplings in range(1, 4):
            self.assertSequenceEqual(
                shape_logic.calculate_post_upsampling_crops(
                    upsamplings=upsamplings
                ),
                (0, 0, 0)
            )

    def test_output_padding_on_correct_side(self) -> None:
        """Test transposed convolution adds padding on correct side"""
        test_input = ones(2, 3, 4)
        test_output = conv_transpose1d(
            test_input,
            ones(3, 3, 2),
            stride=2,
            output_padding=(1,))
        self.assertSequenceEqual(test_output.shape, (2, 3, 9))
        assert_close(
            test_output[:, :, -1],
            zeros_like(test_output[:, :, -1])
        )
        test_input = ones(2, 3, 4, 5)
        test_output = conv_transpose2d(
            test_input,
            ones(3, 3, 2, 2),
            stride=2,
            output_padding=(1, 1))
        self.assertSequenceEqual(test_output.shape, (2, 3, 9, 11))
        assert_close(
            test_output[:, :, -1],
            zeros_like(test_output[:, :, -1])
        )
        assert_close(
            test_output[:, :, :, -1],
            zeros_like(test_output[:, :, :, -1])
        )
        test_input = ones(2, 3, 4, 4, 5)
        test_output = conv_transpose3d(
            test_input,
            ones(3, 3, 2, 2, 2),
            stride=2,
            output_padding=(1, 1, 1))
        self.assertSequenceEqual(test_output.shape, (2, 3, 9, 9, 11))
        assert_close(
            test_output[:, :, -1],
            zeros_like(test_output[:, :, -1])
        )
        assert_close(
            test_output[:, :, :, -1],
            zeros_like(test_output[:, :, :, -1])
        )
        assert_close(
            test_output[:, :, :, :, -1],
            zeros_like(test_output[:, :, :, :, -1])
        )

