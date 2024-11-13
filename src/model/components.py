"""Useful components for building neural networks"""

from abc import abstractmethod
from types import EllipsisType
from typing import Sequence, Union

from torch import Tensor, cat
from torch.nn import Module, ModuleList

from model.activation import get_activation_factory
from model.interface import IActivationFactory, INormalizerFactory
from model.normalizer import get_normalizer_factory
from util.ndimensional_operators import avg_pool_nd, conv_nd, conv_transpose_nd


class ConvNd(Module):
    """Single convolution"""

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        kernel_size: Sequence[int],
        padding: int,
        bias: bool = True,
        stride: Sequence[int] | int = 1,
    ) -> None:
        super().__init__()
        self.conv = conv_nd(len(kernel_size))(
            n_input_channels,
            n_output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            stride=stride,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Double convolve 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, *spatial_shape)

        Returns:
            Tensor with shape (batch_size, n_output_channels, *output_spatial_shape)
        """
        return self.conv(input_tensor)


class ConvBlockNd(Module):
    """Multiple convolutions"""

    def __init__(
        self,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        kernel_size: Sequence[int],
        padding: int,
        activation_factory: IActivationFactory | None = None,
        normalizer_factory: INormalizerFactory | None = None,
    ) -> None:
        super().__init__()
        normalizer_factory = get_normalizer_factory(normalizer_factory)
        activation_factory = get_activation_factory(activation_factory)
        self.convolutions = ModuleList()
        self.convolutions.append(
            ConvNd(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )
        for _ in range(1, n_convolutions):
            self.convolutions.append(
                ConvNd(
                    n_output_channels,
                    n_output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
        self.normalizers = ModuleList()
        self.activations = ModuleList()
        for _ in range(n_convolutions):
            self.normalizers.append(normalizer_factory.build(n_output_channels))
            self.activations.append(activation_factory.build())

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Double convolve 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width, height)
        """
        output = input_tensor
        for convolution, activation, normalizer in zip(
            self.convolutions, self.activations, self.normalizers
        ):
            output = convolution(output)
            output = normalizer(output)
            output = activation(output)
        return output


class _BaseDownsamplingBlockNd(Module):
    """Base downsampling block"""

    def __init__(
        self,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        kernel_size: Sequence[int],
        activation_factory: IActivationFactory | None = None,
        normalizer_factory: INormalizerFactory | None = None,
        downsampling_factor: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        n_dims = len(kernel_size)
        if downsampling_factor is None:
            downsampling_factor = [2] * n_dims
        self.downsampling = self._get_downsampling_layer(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            downsampling_factor=downsampling_factor,
        )
        self.conv = ConvBlockNd(
            n_convolutions=n_convolutions,
            n_input_channels=n_output_channels,
            n_output_channels=n_output_channels,
            kernel_size=kernel_size,
            padding=1,
            activation_factory=activation_factory,
            normalizer_factory=normalizer_factory,
        )

    @abstractmethod
    def _get_downsampling_layer(
        self,
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
        output = self.conv(output)
        return output


class ConvDownsamplingBlockNd(_BaseDownsamplingBlockNd):
    """Convolution downsampling block"""

    def _get_downsampling_layer(
        self,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        return ConvNd(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            kernel_size=downsampling_factor,
            padding=0,
            bias=True,
            stride=downsampling_factor,
        )


class _BaseDownsamplingBlockWithSkipNd(Module):
    """Base downsampling block with skip connection"""

    def __init__(
        self,
        n_convolutions: int,
        n_input_channels: int,
        n_output_channels: int,
        kernel_size: Sequence[int],
        activation_factory: IActivationFactory | None = None,
        normalizer_factory: INormalizerFactory | None = None,
        downsampling_factor: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        n_dims = len(kernel_size)
        if downsampling_factor is None:
            downsampling_factor = [2] * n_dims
        self.downsampling = self._get_downsampling_layer(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            downsampling_factor=downsampling_factor,
        )
        self.conv = ConvBlockNd(
            n_convolutions=n_convolutions,
            n_input_channels=n_output_channels,
            n_output_channels=n_output_channels,
            kernel_size=kernel_size,
            padding=1,
            activation_factory=activation_factory,
            normalizer_factory=normalizer_factory,
        )
        self.average_pool = avg_pool_nd(n_dims)(kernel_size=downsampling_factor)
        self.projection = ConvNd(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            kernel_size=(1,) * n_dims,
            padding=0,
            bias=True,
        )

    @abstractmethod
    def _get_downsampling_layer(
        self,
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
        main = self.conv(main)
        return main, next_skip


class ConvDownsamplingBlockWithSkipNd(_BaseDownsamplingBlockWithSkipNd):
    """Convolution downsampling block with skip connection"""

    def _get_downsampling_layer(
        self,
        n_input_channels: int,
        n_output_channels: int,
        downsampling_factor: Sequence[int],
    ) -> Module:
        return ConvNd(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            kernel_size=downsampling_factor,
            padding=0,
            bias=True,
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
        output_paddings: Sequence[int],
        cropping_slice: tuple[Union[EllipsisType, slice], ...],
        activation_factory: IActivationFactory | None = None,
        normalizer_factory: INormalizerFactory | None = None,
    ) -> None:
        super().__init__()
        self.upsampling = conv_transpose_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_upsampled_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_paddings,
        )
        self.conv = ConvBlockNd(
            n_convolutions=n_convolutions,
            n_input_channels=n_upsampled_channels + n_skip_channels,
            n_output_channels=n_output_channels,
            kernel_size=(3,) * n_dims,
            padding=1,
            activation_factory=activation_factory,
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
