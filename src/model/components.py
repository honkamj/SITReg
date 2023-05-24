"""Useful components for building neural networks"""

from abc import abstractmethod
from types import EllipsisType
from typing import Callable, Optional, Sequence, Union

from torch import Tensor, cat
from torch.nn import Module, ModuleList

from model.interface import INormalizerFactory
from model.normalizer import EmptyNormalizerFactory
from util.ndimensional_operators import avg_pool_nd, conv_nd, conv_transpose_nd


class ConvolutionBlockNd(Module):
    """Double convolution"""

    def __init__(
        self,
        n_dims: int,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        activation: Callable[[Tensor], Tensor],
        normalizer_factory: Optional[INormalizerFactory] = None,
    ) -> None:
        super().__init__()
        normalizer_factory = (
            EmptyNormalizerFactory() if normalizer_factory is None else normalizer_factory
        )
        self.activation = activation
        self.convolutions = ModuleList()
        self.convolutions.append(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        for _ in range(1, n_convolutions):
            self.convolutions.append(
                conv_nd(n_dims)(
                    in_channels=n_output_channels,
                    out_channels=n_output_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.normalizer_modules = ModuleList()
        self._normalizers = []
        for _ in range(n_convolutions):
            normalizer = normalizer_factory.build(n_output_channels)
            if isinstance(normalizer, Module):
                self.normalizer_modules.append(normalizer)
            self._normalizers.append(normalizer)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Double convolve 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width, height)
        """
        output = input_tensor
        for convolution, normalizer in zip(self.convolutions, self._normalizers):
            output = convolution(output)
            output = self.activation(output)
            output = normalizer(output)
        return output


class _BaseDownsamplingBlockNd(Module):
    """Base downsampling block"""

    def __init__(
        self,
        n_dims: int,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        activation: Callable[[Tensor], Tensor],
        normalizer_factory: Optional[INormalizerFactory] = None,
        downsampling_factor: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if downsampling_factor is None:
            downsampling_factor = [2] * n_dims
        normalizer_factory = (
            EmptyNormalizerFactory() if normalizer_factory is None else normalizer_factory
        )
        self.downsampling = self._get_downsampling_layer(
            n_dims, n_input_channels, n_output_channels, downsampling_factor
        )
        self.double_conv = ConvolutionBlockNd(
            n_dims=n_dims,
            n_convolutions=n_convolutions,
            n_input_channels=n_output_channels,
            n_output_channels=n_output_channels,
            activation=activation,
            normalizer_factory=normalizer_factory,
        )

    @abstractmethod
    def _get_downsampling_layer(
        self,
        n_dims: int,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        """Get downsampling layer"""

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Downsampling block forward

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width // 2, height // 2)
        """
        output = self.downsampling(input_tensor)
        output = self.double_conv(output)
        return output


class ConvDownsamplingBlockNd(_BaseDownsamplingBlockNd):
    """Convolution downsampling block"""

    def _get_downsampling_layer(
        self,
        n_dims: int,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        return conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=downsampling_factor,
            stride=downsampling_factor,
        )


class _BaseDownsamplingBlockWithSkipNd(Module):
    """Base downsampling block with skip connection"""

    def __init__(
        self,
        n_dims: int,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        activation: Callable[[Tensor], Tensor],
        normalizer_factory: Optional[INormalizerFactory] = None,
        downsampling_factor: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if downsampling_factor is None:
            downsampling_factor = [2] * n_dims
        normalizer_factory = (
            EmptyNormalizerFactory() if normalizer_factory is None else normalizer_factory
        )
        self.downsampling = self._get_downsampling_layer(
            n_dims, n_input_channels, n_output_channels, downsampling_factor
        )
        self.double_conv = ConvolutionBlockNd(
            n_dims=n_dims,
            n_convolutions=n_convolutions,
            n_input_channels=n_output_channels,
            n_output_channels=n_output_channels,
            activation=activation,
            normalizer_factory=normalizer_factory,
        )
        self.average_pool = avg_pool_nd(n_dims)(kernel_size=2)
        self.projection = conv_nd(n_dims)(
            in_channels=n_input_channels, out_channels=n_output_channels, kernel_size=1, padding=0
        )

    @abstractmethod
    def _get_downsampling_layer(
        self,
        n_dims: int,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        """Get downsampling layer"""

    def forward(self, input_tensor: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
        """Downsampling block forward

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width // 2, height // 2)
        """
        main = self.downsampling(input_tensor)
        main = self.projection(self.average_pool(skip)) + main
        next_skip = main
        main = self.double_conv(main)
        return main, next_skip


class ConvDownsamplingBlockWithSkipNd(_BaseDownsamplingBlockWithSkipNd):
    """Convolution downsampling block with skip connection"""

    def _get_downsampling_layer(
        self,
        n_dims: int,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        return conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=downsampling_factor,
            stride=downsampling_factor,
        )


class ConvTransposedUpsamplingBlockNd(Module):
    """Upsampling block with skip connection from downsampling path"""

    def __init__(
        self,
        n_dims: int,
        n_convolutions: int,
        n_input_channels: int,
        n_skip_channels: int,
        n_upsampled_channels: int,
        n_output_channels: int,
        activation: Callable[[Tensor], Tensor],
        output_paddings: Sequence[int],
        cropping_slice: tuple[Union[EllipsisType, slice], ...],
        normalizer_factory: Optional[INormalizerFactory] = None,
    ) -> None:
        super().__init__()
        normalizer_factory = (
            EmptyNormalizerFactory() if normalizer_factory is None else normalizer_factory
        )
        self.upsampling = conv_transpose_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_upsampled_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_paddings,
        )
        self.conv = ConvolutionBlockNd(
            n_dims=n_dims,
            n_convolutions=n_convolutions,
            n_input_channels=n_upsampled_channels + n_skip_channels,
            n_output_channels=n_output_channels,
            activation=activation,
            normalizer_factory=normalizer_factory,
        )
        self._cropping_slice = cropping_slice

    def forward(self, input_tensor: Tensor, skip_tensor: Tensor) -> Tensor:
        """Upsampling block forward

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)
            skip_tensor: Tensor with shape (batch_size, n_skip_channels, 2 * width, 2 * height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, 2 * width, 2 * height)
        """
        upsampled = self.upsampling(input_tensor)[self._cropping_slice]
        concatenated = cat([upsampled, skip_tensor], dim=1)
        output = self.conv(concatenated)
        return output
