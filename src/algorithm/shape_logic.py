"""Shape logic for different kinds of networks"""

from math import ceil, floor
from types import EllipsisType
from typing import Sequence, Union

from torch import Tensor
from torch.nn.functional import pad


class EncoderShapeLogic:
    """Shape logic of an encoder extracting multi-level features

    Arguments:
        shape_mode: Either "floor" or "ceil". At each stage resolutions
            is halved. This determines how it is done if shape is not
            divisible by 2.
        n_feature_levels: Essentially amount of items in n_features_per_resolution
            list.
        input_shape: Input shape to the network
        downsampling_factor: Downsampling factor between feature levels.
            If Sequence is given, corresponds to different downsampling factors
            between different levels and if Sequence of Sequences is given
            corresponds additionally to different downsampling factors over
            different dimensions.
    """

    def __init__(
        self,
        shape_mode: str,
        n_feature_levels: int,
        input_shape: Sequence[int],
        downsampling_factor: int | Sequence[int | Sequence[int]],
    ) -> None:
        self._shape_mode = shape_mode
        self._shape_function = ceil if shape_mode == "ceil" else floor
        self._n_feature_levels = n_feature_levels
        self._input_shape = input_shape
        if isinstance(downsampling_factor, int):
            self._downsampling_factors: Sequence[Sequence[int]] = [
                [downsampling_factor] * len(input_shape)
            ] * (n_feature_levels - 1)
        else:  # Sequence[int | Sequence[int]]
            assert len(downsampling_factor) == n_feature_levels - 1
            self._downsampling_factors = []
            for factor in downsampling_factor:
                if isinstance(factor, int):
                    self._downsampling_factors.append([factor] * len(input_shape))
                else:  # Sequence[int]
                    self._downsampling_factors.append(factor)

    @property
    def n_feature_levels(self) -> int:
        """Number of feature levels"""
        return self._n_feature_levels

    def calculate_downsampling_factor(self, downsamplings: int) -> Sequence[float]:
        """Calculate combined downsampling factor after given amount of downsamplings"""
        assert downsamplings >= 0 and downsamplings < self._n_feature_levels
        if downsamplings > 0:
            earlier_factor = self.calculate_downsampling_factor(downsamplings - 1)
            current_factor = self._downsampling_factors[downsamplings - 1]
            return [
                dim_earlier_factor * dim_current_factor
                for dim_earlier_factor, dim_current_factor in zip(earlier_factor, current_factor)
            ]
        return [1.0] * len(self._input_shape)

    def calculate_downsampling_factors(self) -> Sequence[Sequence[float]]:
        """Calculate combined downsampling factors at different levels"""
        return [
            self.calculate_downsampling_factor(downsamplings)
            for downsamplings in range(self._n_feature_levels)
        ]

    def calculate_shape(self, downsamplings: int) -> Sequence[int]:
        """Calculate shape after given amount of downsamplings"""
        downsampling_factor = self.calculate_downsampling_factor(downsamplings)
        return [
            int(self._shape_function(dim_size / dim_downsampling_factor))
            for dim_size, dim_downsampling_factor in zip(self._input_shape, downsampling_factor)
        ]

    def calculate_shapes(self) -> Sequence[Sequence[int]]:
        """Calculate shapes at different levels"""
        return [
            self.calculate_shape(downsamplings=downsamplings)
            for downsamplings in range(self._n_feature_levels)
        ]

    @staticmethod
    def _convert_right_paddings_to_pad_paddings(right_paddings: Sequence[int]) -> tuple[int, ...]:
        flattened_paddings: tuple[int, ...] = tuple()
        for right_padding in reversed(right_paddings):
            flattened_paddings += (0, right_padding)
        return flattened_paddings

    def calculate_pre_downsampling_paddings(self, earlier_downsamplings: int) -> Sequence[int]:
        """Paddings added before a downsampling to the right along each dimension"""
        assert earlier_downsamplings >= 0 and earlier_downsamplings < self._n_feature_levels - 1
        if self._shape_mode == "floor":
            return [0] * len(self._input_shape)
        return [
            dim_size % dim_downsampling_factor
            for dim_size, dim_downsampling_factor in zip(
                self.calculate_shape(earlier_downsamplings),
                self._downsampling_factors[earlier_downsamplings],
            )
        ]

    def pre_downsampling_pad(self, tensor: Tensor, earlier_downsamplings: int) -> Tensor:
        """Apply padding before downsampling of given index"""
        return pad(
            input=tensor,
            pad=self._convert_right_paddings_to_pad_paddings(
                self.calculate_pre_downsampling_paddings(
                    earlier_downsamplings=earlier_downsamplings
                )
            ),
            mode="constant",
        )


class DecoderShapeLogic:
    """Shape logic  of a decoder corresponding to some encoder

    Arguments:
        encoder_shape_logic: Shape logic for encoder
    """

    def __init__(self, encoder_shape_logic: EncoderShapeLogic) -> None:
        self._encoder_shape_logic = encoder_shape_logic

    def upsampling_output_padding(self, earlier_upsamplings: int) -> tuple[int, ...]:
        """Output paddings added in transposed convolution along each dimesion"""
        assert (
            earlier_upsamplings >= 0
            and earlier_upsamplings < self._encoder_shape_logic.n_feature_levels - 1
        )
        current_downsamplings = self._encoder_shape_logic.n_feature_levels - (
            earlier_upsamplings + 1
        )
        current_shape = self._encoder_shape_logic.calculate_shape(
            downsamplings=current_downsamplings
        )
        target_shape = self._encoder_shape_logic.calculate_shape(
            downsamplings=current_downsamplings - 1
        )
        return tuple(
            max(target_dim_size - 2 * current_dim_size, 0)
            for target_dim_size, current_dim_size in zip(target_shape, current_shape)
        )

    def calculate_post_upsampling_crops(self, upsamplings: int) -> Sequence[int]:
        """Cropping done from the right after an upsampling step"""
        assert upsamplings >= 1 and upsamplings < self._encoder_shape_logic.n_feature_levels
        return self._encoder_shape_logic.calculate_pre_downsampling_paddings(
            earlier_downsamplings=self._encoder_shape_logic.n_feature_levels - (upsamplings + 1)
        )

    def generate_post_upsampling_crop(
        self, upsamplings: int
    ) -> tuple[Union[EllipsisType, slice], ...]:
        """Crop as a slice tuple applied after an upsampling

        Spatial dimensions are assumed to be last.
        """
        slice_tuple = tuple(
            slice(0, -dim_crop if dim_crop > 0 else None)
            for dim_crop in self.calculate_post_upsampling_crops(upsamplings=upsamplings)
        )
        return (...,) + slice_tuple
