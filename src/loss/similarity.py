"""MSE loss"""

from math import floor
from typing import Sequence

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, matmul, ones, prod, tensor

from algorithm.composable_mapping.interface import IMaskedTensor
from algorithm.composable_mapping.masked_tensor import MaskedTensor
from loss.interface import ISimilarityLoss
from util.ndimensional_operators import avg_pool_nd, conv_nd_function
from util.optional import optional_add


class MeanSquaredError(ISimilarityLoss):
    """MSE loss for mappings"""

    def __call__(
        self,
        image_1: IMaskedTensor,
        image_2: IMaskedTensor,
        device: torch_device | None = None,
        dtype: torch_dtype | None = None,
    ) -> Tensor:
        image_1_values = image_1.generate_values(device, dtype)
        image_2_values = image_1.generate_values(device, dtype)
        return (
            (image_1_values - image_2_values).square()
            * image_1.generate_mask(device, dtype)
            * image_2.generate_mask(device, dtype)
        ).mean()


class LocalNormalizedCrossCorrelationLoss(ISimilarityLoss):
    """Local normalized cross correlation loss

    Modified from https://github.com/voxelmorph.
    """

    def __init__(self, window_size: int, epsilon: float = 1e-5):
        self._window_size = window_size
        self._epsilon = epsilon

    def __call__(
        self,
        image_1: IMaskedTensor,
        image_2: IMaskedTensor,
        device: torch_device | None = None,
        dtype: torch_dtype | None = None,
    ) -> Tensor:
        """Compute the loss"""
        source_volume, target_volume = _multiply_both_by_mask_union(image_1, image_2, device, dtype)
        n_dims = source_volume.ndim - 2
        if n_dims not in [1, 2, 3]:
            raise ValueError(f"Volumes should have 1 to 3 dimensions, found {n_dims}")
        window_shape = [self._window_size] * n_dims
        sum_filter = ones(1, 1, *window_shape, device=device, dtype=dtype)
        stride = (1,) * n_dims
        paddings = (floor(self._window_size / 2),) * n_dims
        conv_function = conv_nd_function(n_dims)

        source_sums = conv_function(source_volume, sum_filter, stride=stride, padding=paddings)
        target_sums = conv_function(target_volume, sum_filter, stride=stride, padding=paddings)
        source_square_sums = conv_function(
            source_volume.square(), sum_filter, stride=stride, padding=paddings
        )
        target_square_sums = conv_function(
            target_volume.square(), sum_filter, stride=stride, padding=paddings
        )
        cross_sums = conv_function(
            source_volume * target_volume, sum_filter, stride=stride, padding=paddings
        )

        window_size = prod(tensor(window_shape, device=device, dtype=dtype))
        source_means = source_sums / window_size
        target_means = target_sums / window_size

        cross = (
            cross_sums
            - target_means * source_sums
            - source_means * target_sums
            + source_means * target_means * window_size
        )
        source_variance = (
            source_square_sums
            - 2 * source_means * source_sums
            + source_means.square() * window_size
        )
        target_variance = (
            target_square_sums
            - 2 * target_means * target_sums
            + target_means.square() * window_size
        )

        cross_correlation = cross * cross / (source_variance * target_variance + self._epsilon)

        return -cross_correlation.mean()


class MultiResolutionLocalNormalizedCrossCorrelationLoss(ISimilarityLoss):
    """Multi-resolution local normalized cross correlation loss

    Mok, Tony CW, and Albert Chung. "Large deformation diffeomorphic image
    registration with laplacian pyramid networks." (2020)

    Modified from https://github.com/cwmok/Conditional_LapIRN/

    Mok et al. uses kernel size 3 for downsampling whereas we use the same
    kernel size and stride.
    """

    def __init__(
        self,
        n_dims: int,
        window_sizes_per_resolution: Sequence[int],
        weights_per_resolution: Sequence[float],
        downsampling_factors_per_resolution: Sequence[int],
        downsampling_kernel_sizes_per_resolution: Sequence[int],
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self._n_dims = n_dims
        self._weights_per_resolution = weights_per_resolution
        self._losses: list[LocalNormalizedCrossCorrelationLoss] = [
            LocalNormalizedCrossCorrelationLoss(
                window_size=window_size,
                epsilon=epsilon,
            )
            for window_size in window_sizes_per_resolution
        ]
        self._avg_pools = [
            avg_pool_nd(self._n_dims)(
                kernel_size=kernel_size,
                stride=downsampling_factor,
            )
            for downsampling_factor, kernel_size in zip(downsampling_factors_per_resolution, downsampling_kernel_sizes_per_resolution)
        ]

    def __call__(
        self,
        image_1: IMaskedTensor,
        image_2: IMaskedTensor,
        device: torch_device | None = None,
        dtype: torch_dtype | None = None,
    ) -> Tensor:
        source_volume = image_1.generate_values(device, dtype)
        target_volume = image_2.generate_values(device, dtype)
        device = source_volume.device
        dtype = source_volume.dtype
        if image_1.has_mask():
            source_volume = source_volume * image_1.generate_mask(device, dtype)
            target_volume = target_volume * image_1.generate_mask(device, dtype)
        if image_2.has_mask():
            source_volume = source_volume * image_2.generate_mask(device, dtype)
            target_volume = target_volume * image_2.generate_mask(device, dtype)
        loss: Tensor | None = None
        for loss_function, avg_pool, weight in zip(
            self._losses, self._avg_pools, self._weights_per_resolution
        ):
            source_volume = avg_pool(source_volume)
            target_volume = avg_pool(target_volume)
            resolution_loss = loss_function(
                MaskedTensor(source_volume),
                MaskedTensor(target_volume),
                device=device,
                dtype=dtype,
            )
            loss = optional_add(loss, resolution_loss * weight)
        assert loss is not None
        return loss


class MutualInformationLoss(ISimilarityLoss):
    """Differentiable global mutual information via Parzen windowing method.

    Guo, Courtney K. Multi-modal image registration with unsupervised deep
    learning. Diss. Massachusetts Institute of Technology, 2019.

    Rewritten from https://github.com/DeepRegNet/DeepReg

    The inputs are clipped to range [0, 1] after applying shift and normalization.
    """

    def __init__(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        shift: float = 0.0,
        normalization: float = 1.0,
        epsilon: float = 1e-5,
    ) -> None:
        self._num_bins = num_bins
        self._sigma_ratio = sigma_ratio
        self._shift = shift
        self._normalization = normalization
        self._epsilon = epsilon

    def _pre_process(self, image: Tensor) -> Tensor:
        return ((image - self._shift) / self._normalization).clamp(0, 1)

    def __call__(
        self,
        image_1: IMaskedTensor,
        image_2: IMaskedTensor,
        device: torch_device | None = None,
        dtype: torch_dtype | None = None,
    ) -> Tensor:
        volume_1, volume_2 = _multiply_both_by_mask_union(image_1, image_2, device, dtype)
        dtype = volume_1.dtype
        device = volume_1.device
        batch_size = volume_1.size(0)

        volume_1 = self._pre_process(volume_1)
        volume_2 = self._pre_process(volume_2)
        bin_centers = linspace(0.0, 1.0, self._num_bins, dtype=dtype, device=device)
        sigma = (bin_centers[1:] - bin_centers[:-1]).mean() * self._sigma_ratio
        preterm = 1 / (2 * sigma.square())
        volume_1 = volume_1.view(batch_size, -1, 1)
        volume_2 = volume_2.view(batch_size, -1, 1)
        n_values = float(volume_1.size(1))

        voxel_densities_1, marginal_density_1 = self._compute_densities(
            volume_1, bin_centers=bin_centers, preterm=preterm
        )
        voxel_densities_2, marginal_density_2 = self._compute_densities(
            volume_2, bin_centers=bin_centers, preterm=preterm
        )

        product_density = matmul(marginal_density_1[..., None], marginal_density_2[..., None, :])
        joint_density = matmul(voxel_densities_1.transpose(1, 2), voxel_densities_2) / n_values

        density_ratio = (joint_density + self._epsilon) / (product_density + self._epsilon)
        return -(joint_density * (density_ratio + self._epsilon).log()).sum(dim=(1, 2)).mean()

    @staticmethod
    def _compute_densities(
        volume: Tensor, bin_centers: Tensor, preterm: Tensor
    ) -> tuple[Tensor, Tensor]:
        volume = volume.view(volume.size(0), -1, 1)
        voxel_densities_unnormalized = (
            -preterm * (volume - bin_centers[None, None]).square()
        ).exp()
        voxel_densities = voxel_densities_unnormalized / voxel_densities_unnormalized.sum(
            dim=-1, keepdim=True
        )
        marginal_density = voxel_densities.mean(1)
        return voxel_densities, marginal_density


def _multiply_both_by_mask_union(
    image_1: IMaskedTensor,
    image_2: IMaskedTensor,
    device: torch_device | None,
    dtype: torch_dtype | None,
) -> tuple[Tensor, Tensor]:
    """Multiply both images by mask union and return them"""
    volume_1 = image_1.generate_values(device, dtype)
    volume_2 = image_2.generate_values(device, dtype)
    device = volume_1.device
    dtype = volume_1.dtype
    if image_1.has_mask():
        volume_1 = volume_1 * image_1.generate_mask(device, dtype)
        volume_2 = volume_2 * image_1.generate_mask(device, dtype)
    if image_2.has_mask():
        volume_1 = volume_1 * image_2.generate_mask(device, dtype)
        volume_2 = volume_2 * image_2.generate_mask(device, dtype)
    return volume_1, volume_2
