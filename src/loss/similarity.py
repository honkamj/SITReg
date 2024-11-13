"""MSE loss"""

from math import floor, prod
from typing import Any, Mapping, Sequence

from composable_mapping import MappableTensor, mappable
from torch import Tensor
from torch import linspace, matmul

from algorithm.sum_pool import sum_pool_nd
from loss.interface import ISimilarityLoss
from util.dimension_order import get_other_than_batch_dim
from util.ndimensional_operators import avg_pool_nd
from util.optional import optional_add

from .util import build_default_params, handle_params


class MeanSquaredError(ISimilarityLoss):
    """MSE loss for mappings"""

    def __call__(
        self,
        image_1: MappableTensor,
        image_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        params = handle_params(params, {"weight": 1.0})
        image_1_values = image_1.generate_values()
        image_2_values = image_1.generate_values()
        return (
            params["weight"]
            * (
                (image_1_values - image_2_values).square()
                * image_1.generate_mask(generate_missing_mask=True, cast_mask=True)
                * image_2.generate_mask(generate_missing_mask=True, cast_mask=True)
            ).mean()
        )


class LocalNormalizedCrossCorrelationLoss(ISimilarityLoss):
    """Local normalized cross correlation loss

    Modified from https://github.com/voxelmorph.
    """

    def __init__(
        self, window_size: int | None, epsilon: float | None = 1e-5, separable: bool = False
    ):
        self._default_params = build_default_params(
            none_ignored_params={
                "window_size": window_size,
                "epsilon": epsilon,
                "weight": 1.0,
                "separable": separable,
            }
        )

    def __call__(
        self,
        image_1: MappableTensor,
        image_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Compute the loss"""
        params = handle_params(params, self._default_params)
        source_volume, target_volume = _multiply_both_by_mask_union(image_1, image_2)
        n_dims = source_volume.ndim - 2
        if n_dims not in [1, 2, 3]:
            raise ValueError(f"Volumes should have 1 to 3 dimensions, found {n_dims}")
        window_shape = [params["window_size"]] * n_dims
        stride = (1,) * n_dims
        padding = (floor(params["window_size"] / 2),) * n_dims

        source_sums = sum_pool_nd(
            source_volume,
            kernel_size=window_shape,
            stride=stride,
            padding=padding,
            separable=params["separable"],
        )
        target_sums = sum_pool_nd(
            target_volume,
            kernel_size=window_shape,
            stride=stride,
            padding=padding,
            separable=params["separable"],
        )
        source_square_sums = sum_pool_nd(
            source_volume.square(),
            kernel_size=window_shape,
            stride=stride,
            padding=padding,
            separable=params["separable"],
        )
        target_square_sums = sum_pool_nd(
            target_volume.square(),
            kernel_size=window_shape,
            stride=stride,
            padding=padding,
            separable=params["separable"],
        )
        cross_sums = sum_pool_nd(
            source_volume * target_volume,
            kernel_size=window_shape,
            stride=stride,
            padding=padding,
            separable=params["separable"],
        )

        window_size = prod(window_shape)
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
        epsilon: float | int = params["epsilon"]
        weight: float | int | Tensor = params["weight"]
        cross_correlation = cross * cross / (source_variance * target_variance + epsilon)
        return (
            -cross_correlation.mean(dim=get_other_than_batch_dim(cross_correlation)) * weight
        ).mean()


class MultiResolutionLocalNormalizedCrossCorrelationLoss(ISimilarityLoss):
    """Multi-resolution local normalized cross correlation loss

    Mok, Tony CW, and Albert Chung. "Large deformation diffeomorphic image
    registration with laplacian pyramid networks." (2020)

    Modified from https://github.com/cwmok/Conditional_LapIRN
    """

    def __init__(
        self,
        n_dims: int,
        window_sizes_per_resolution: Sequence[int] | None,
        weights_per_resolution: Sequence[float] | None,
        downsampling_factors_per_resolution: Sequence[int],
        downsampling_kernel_sizes_per_resolution: Sequence[int],
        separable_per_resolution: Sequence[bool],
        epsilon: float | None = 1e-5,
    ):
        super().__init__()
        self._n_dims = n_dims
        self._losses: list[LocalNormalizedCrossCorrelationLoss] = [
            LocalNormalizedCrossCorrelationLoss(
                window_size=None,
                epsilon=None,
            )
            for _ in range(len(downsampling_factors_per_resolution))
        ]
        self._default_params = build_default_params(
            none_ignored_params={
                "window_sizes_per_resolution": window_sizes_per_resolution,
                "weights_per_resolution": weights_per_resolution,
                "separable_per_resolution": separable_per_resolution,
                "epsilon": epsilon,
                "weight": 1.0,
            }
        )
        self._avg_pools = [
            avg_pool_nd(self._n_dims)(
                kernel_size=kernel_size,
                stride=downsampling_factor,
            )
            for downsampling_factor, kernel_size in zip(
                downsampling_factors_per_resolution,
                downsampling_kernel_sizes_per_resolution,
            )
        ]

    def __call__(
        self,
        image_1: MappableTensor,
        image_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        params = handle_params(params, self._default_params)
        source_volume = image_1.generate_values()
        target_volume = image_2.generate_values()
        if image_1.has_mask():
            source_volume = source_volume * image_1.generate_mask(
                cast_mask=True, generate_missing_mask=True
            )
            target_volume = target_volume * image_1.generate_mask(
                cast_mask=True, generate_missing_mask=True
            )
        if image_2.has_mask():
            source_volume = source_volume * image_2.generate_mask(
                cast_mask=True, generate_missing_mask=True
            )
            target_volume = target_volume * image_2.generate_mask(
                cast_mask=True, generate_missing_mask=True
            )
        loss: Tensor | None = None
        for loss_function, avg_pool, weight, window_size, separable in zip(
            self._losses,
            self._avg_pools,
            params["weights_per_resolution"],
            params["window_sizes_per_resolution"],
            params["separable_per_resolution"],
        ):
            source_volume = avg_pool(source_volume)
            target_volume = avg_pool(target_volume)
            resolution_loss = loss_function(
                mappable(source_volume),
                mappable(target_volume),
                params={
                    "epsilon": params["epsilon"],
                    "window_size": window_size,
                    "weight": weight * params["weight"],
                    "separable": separable,
                },
            )
            loss = optional_add(loss, resolution_loss)
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
        num_bins: int | None = 23,
        sigma_ratio: float | None = 0.5,
        shift: float | None = 0.0,
        normalization: float | None = 1.0,
        epsilon: float | None = 1e-5,
    ) -> None:
        self._default_params = build_default_params(
            none_ignored_params={
                "num_bins": num_bins,
                "sigma_ratio": sigma_ratio,
                "shift": shift,
                "normalization": normalization,
                "epsilon": epsilon,
            }
        )

    def _pre_process(self, image: Tensor, params: Mapping[str, Any]) -> Tensor:
        return ((image - params["shift"]) / params["normalization"]).clamp(0, 1)

    def __call__(
        self,
        image_1: MappableTensor,
        image_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        params = handle_params(params, self._default_params)
        volume_1, volume_2 = _multiply_both_by_mask_union(image_1, image_2)
        dtype = volume_1.dtype
        device = volume_1.device
        batch_size = volume_1.size(0)

        volume_1 = self._pre_process(volume_1, params)
        volume_2 = self._pre_process(volume_2, params)
        bin_centers = linspace(0.0, 1.0, params["num_bins"], dtype=dtype, device=device)
        sigma = (bin_centers[1:] - bin_centers[:-1]).mean() * params["sigma_ratio"]
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

        density_ratio = (joint_density + params["epsilon"]) / (product_density + params["epsilon"])
        return -(joint_density * (density_ratio + params["epsilon"]).log()).sum(dim=(1, 2)).mean()

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
    image_1: MappableTensor,
    image_2: MappableTensor,
) -> tuple[Tensor, Tensor]:
    """Multiply both images by mask union and return them"""
    volume_1 = image_1.generate_values()
    volume_2 = image_2.generate_values()
    if image_1.has_mask():
        volume_1 = volume_1 * image_1.generate_mask(generate_missing_mask=True, cast_mask=True)
        volume_2 = volume_2 * image_1.generate_mask(generate_missing_mask=True, cast_mask=True)
    if image_2.has_mask():
        volume_1 = volume_1 * image_2.generate_mask(generate_missing_mask=True, cast_mask=True)
        volume_2 = volume_2 * image_2.generate_mask(generate_missing_mask=True, cast_mask=True)
    return volume_1, volume_2
