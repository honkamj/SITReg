"""SITReg feature extractor implementations"""

from typing import Optional, Sequence

from torch import Tensor, cat
from torch.nn import ModuleList

from algorithm.shape_logic import EncoderShapeLogic
from model.components import ConvBlockNd, ConvDownsamplingBlockWithSkipNd
from model.interface import IActivationFactory, INormalizerFactory
from model.normalizer import get_normalizer_factory
from util.ndimensional_operators import conv_nd

from .base import BaseFeatureExtractor


class EncoderFeatureExtractor(BaseFeatureExtractor):
    """Simple convolutional ResNet-style feature extractor with
    resolution halved at each stage

    Arguments:
        n_input_channels: Number of input channels in the input images
        activation: Activation to use
        n_features_per_resolution: Number of features to output for each resolution
        n_convolutions_per_resolution: Number of convolutions to have for each resolution
        input_shape: Input volume shape
        normalizer_factory: Defines the normalizer to use
    """

    def __init__(
        self,
        n_input_channels: int,
        activation_factory: IActivationFactory,
        n_features_per_resolution: Sequence[int],
        n_convolutions_per_resolution: Sequence[int],
        input_shape: Sequence[int],
        normalizer_factory: Optional[INormalizerFactory] = None,
    ) -> None:
        super().__init__()
        n_dims = len(input_shape)
        normalizer_factory = get_normalizer_factory(normalizer_factory)
        self._n_features_per_resolution = n_features_per_resolution
        self.shape_logic = EncoderShapeLogic(
            shape_mode="ceil",
            n_feature_levels=len(n_features_per_resolution),
            input_shape=input_shape,
            downsampling_factor=2,
        )
        self.initial_projection = conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_features_per_resolution[0],
            kernel_size=1,
            padding=0,
        )
        self.initial_convs = ConvBlockNd(
            n_convolutions=n_convolutions_per_resolution[0],
            n_input_channels=n_features_per_resolution[0],
            n_output_channels=n_features_per_resolution[0],
            kernel_size=(3,) * n_dims,
            padding=1,
            activation_factory=activation_factory,
            normalizer_factory=normalizer_factory,
        )
        self.downsampling_blocks = ModuleList()
        for i in range(len(n_features_per_resolution) - 1):
            self.downsampling_blocks.append(
                ConvDownsamplingBlockWithSkipNd(
                    n_convolutions=n_convolutions_per_resolution[i + 1],
                    n_input_channels=n_features_per_resolution[i],
                    n_output_channels=n_features_per_resolution[i + 1],
                    kernel_size=(3,) * n_dims,
                    activation_factory=activation_factory,
                    normalizer_factory=normalizer_factory,
                    downsampling_factor=[2] * n_dims,
                )
            )

    def get_shapes(self) -> Sequence[Sequence[int]]:
        return [
            [n_features] + list(volume_shape)
            for n_features, volume_shape in zip(
                self._n_features_per_resolution, self.shape_logic.calculate_shapes()
            )
        ]

    def _get_downsampling_factors(self) -> Sequence[Sequence[float]]:
        return self.shape_logic.calculate_downsampling_factors()

    def forward(self, images: Sequence[Tensor]) -> Sequence[Tensor]:
        """Compute the features

        Args:
            images: List of Tensors with shape (batch_size, n_channels, *spatial_shape)
        """
        combined_input = cat(tuple(images), dim=0)
        skip = self.initial_projection(combined_input)
        output = self.initial_convs(skip)
        features: list[Tensor] = [output]
        for i, downsampling_block in enumerate(self.downsampling_blocks):
            padded_output = self.shape_logic.pre_downsampling_pad(
                tensor=output, earlier_downsamplings=i
            )
            padded_skip = self.shape_logic.pre_downsampling_pad(
                tensor=skip, earlier_downsamplings=i
            )
            output, skip = downsampling_block(
                padded_output,
                skip=padded_skip,
            )
            features.append(output)
        return features
