"""Masked tensors"""

from types import EllipsisType
from typing import Optional, Sequence, TypeVar

from torch import Tensor, allclose
from torch import any as torch_any
from torch import device as torch_device
from torch import diag, diagonal
from torch import dtype as torch_dtype
from torch import int32 as torch_int32
from torch import ones
from torch import round as torch_round
from torch import tensor

from util.dimension_order import index_by_channel_dims, reduce_channel_shape_to_ones
from util.tensor_cache import TensorCache

from ..dense_deformation import generate_voxel_coordinate_grid
from .affine import Identity
from .interface import (
    IAffineTransformation,
    ICPUComposableAffineTransformation,
    IMaskedTensor,
    IRegularGridTensor,
)


class MaskedTensor(IMaskedTensor):
    """Masked tensor

    Arguments:
        values: Tensor with shape (batch_size, *channel_dims, *spatial_dims)
        mask: Tensor with shape (batch_size, 1, *spatial_dims)
    """

    def __init__(
        self,
        values: Tensor,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        affine_transformation: Optional[IAffineTransformation] = None,
    ) -> None:
        self._values = values
        self._mask = mask
        self._n_channel_dims = n_channel_dims
        first_channel_dim = index_by_channel_dims(values.ndim, 0, n_channel_dims)
        self._channels_shape = values.shape[
            first_channel_dim : first_channel_dim + n_channel_dims
        ]
        self._affine_transformation: IAffineTransformation = (
            Identity(self._channels_shape[0])
            if affine_transformation is None
            else affine_transformation
        )

    @property
    def mask(self) -> Optional[Tensor]:
        return self._mask

    @property
    def shape(self) -> Sequence[int]:
        return self._values.shape

    def generate_mask(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if self._mask is not None:
            return self._mask
        return ones(
            reduce_channel_shape_to_ones(self._values.shape, self._n_channel_dims),
            device=self._values.device,
            dtype=self._values.dtype,
        )

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        return self._affine_transformation(self._values)

    def modified_copy(self, **kwargs) -> "MaskedTensor":
        """Create modified copy"""
        return MaskedTensor(
            values=kwargs["values"] if "values" in kwargs else self._values,
            mask=kwargs["mask"] if "mask" in kwargs else self._mask,
            n_channel_dims=(
                kwargs["n_channel_dims"]
                if "n_channel_dims" in kwargs
                else self._n_channel_dims
            ),
            affine_transformation=(
                kwargs["affine_transformation"]
                if "affine_transformation" in kwargs
                else self._affine_transformation
            ),
        )

    def apply_affine(
        self, affine_transformation: IAffineTransformation
    ) -> "MaskedTensor":
        return self.modified_copy(
            affine_transformation=affine_transformation.compose_affine(
                self._affine_transformation
            )
        )

    @property
    def channels_shape(self) -> Sequence[int]:
        return self._channels_shape

    def has_mask(self) -> bool:
        return self._mask is not None

    def cache(self) -> "_CachedMaskedTensor":
        return _CachedMaskedTensor(self)

    def clear_mask(self) -> "IMaskedTensor":
        return self.modified_copy(mask=None)

    def detach(self) -> "IMaskedTensor":
        return MaskedTensor(
            values=self._values.detach(),
            mask=None if self._mask is None else self._mask.detach(),
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation.detach(),
        )


T = TypeVar("T", bound="_CachedMaskedTensor")


class _CachedMaskedTensor(IMaskedTensor):
    def __init__(self, masked_tensor: IMaskedTensor) -> None:
        self._masked_tensor = masked_tensor
        self._values_cache = TensorCache(masked_tensor.generate_values)
        self._mask_cache = TensorCache(masked_tensor.generate_mask)
        self._affine_transformation: IAffineTransformation = Identity(
            masked_tensor.channels_shape[0]
        )

    @classmethod
    def _create_with_existing_cache(
        cls: type[T],
        masked_tensor: IMaskedTensor,
        affine_transformation: IAffineTransformation,
        values_cache: TensorCache,
        mask_cache: Optional[TensorCache],
    ) -> T:
        """Create cached masked tensor with existing cache

        Note that setting mask_cache to None will result in clearing the mask cache"""
        cached = cls(masked_tensor)
        cached._affine_transformation = affine_transformation
        cached._values_cache = values_cache
        cached._mask_cache = (
            TensorCache(cached._generate_full_mask)
            if mask_cache is None
            else mask_cache
        )
        return cached

    @property
    def mask(self) -> Optional[Tensor]:
        return self._masked_tensor.mask

    @property
    def shape(self) -> Sequence[int]:
        return self._masked_tensor.shape

    def _generate_full_mask(
        self, device: Optional[torch_device], dtype: Optional[torch_dtype]
    ) -> Tensor:
        if device is None:
            raise RuntimeError("Device is needed!")
        return ones(
            reduce_channel_shape_to_ones(self.shape, len(self.channels_shape)),
            device=device,
            dtype=dtype,
        )

    def generate_mask(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None or dtype is None:
            raise RuntimeError("Device and dtype are needed!")
        return self._mask_cache.get(device=device, dtype=dtype)

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None or dtype is None:
            raise RuntimeError("Device and dtype are needed!")
        return self._affine_transformation(
            self._values_cache.get(device=device, dtype=dtype)
        )

    @property
    def channels_shape(self) -> Sequence[int]:
        return self._masked_tensor.channels_shape

    def apply_affine(
        self, affine_transformation: IAffineTransformation
    ) -> "_CachedMaskedTensor":
        return self._create_with_existing_cache(
            masked_tensor=self._masked_tensor,
            affine_transformation=affine_transformation.compose_affine(
                self._affine_transformation
            ),
            values_cache=self._values_cache,
            mask_cache=self._mask_cache,
        )

    def has_mask(self) -> bool:
        return self._masked_tensor.has_mask()

    def cache(self) -> "_CachedMaskedTensor":
        return _CachedMaskedTensor(self)

    def clear_mask(self) -> "IMaskedTensor":
        return self._create_with_existing_cache(
            masked_tensor=self._masked_tensor,
            affine_transformation=self._affine_transformation,
            values_cache=self._values_cache,
            mask_cache=None,
        )

    def detach(self) -> "_CachedMaskedTensor":
        return self._create_with_existing_cache(
            masked_tensor=self._masked_tensor.detach(),
            affine_transformation=self._affine_transformation.detach(),
            values_cache=self._values_cache.detach(),
            mask_cache=self._mask_cache.detach(),
        )


class BaseVoxelCoordinateGrid(IRegularGridTensor):
    """Base implementation for voxel coordinate grid"""

    REDUCE_TO_SLICE_TOL = 1e-5

    def cache(self) -> "_CachedVoxelCoordinateGrid":
        return _CachedVoxelCoordinateGrid(self)

    def reduce_to_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[tuple[EllipsisType | slice, ...]]:
        """Reduce the grid to slice on target shape, if possible"""
        cpu_composable_affine = self.get_cpu_affine()
        if cpu_composable_affine is None:
            return None
        transformation_matrix = cpu_composable_affine.as_cpu_matrix().squeeze()
        if transformation_matrix.ndim != 2:
            return None
        scale = diagonal(transformation_matrix[:-1, :-1])
        if not allclose(
            diag(scale), transformation_matrix[:-1, :-1], atol=self.REDUCE_TO_SLICE_TOL
        ):
            return None
        translation = transformation_matrix[:-1, -1]
        if (
            torch_any(translation < -self.REDUCE_TO_SLICE_TOL)
            or not allclose(
                translation.round(), translation, atol=self.REDUCE_TO_SLICE_TOL
            )
            or not allclose(scale.round(), scale, atol=self.REDUCE_TO_SLICE_TOL)
        ):
            return None
        scale = torch_round(scale).type(torch_int32)
        translation = torch_round(translation).type(torch_int32)
        target_shape_tensor = tensor(target_shape, dtype=torch_int32)
        shape_tensor = tensor(self.shape, dtype=torch_int32)
        slice_ends = (shape_tensor - 1) * scale + translation + 1
        if torch_any(slice_ends > target_shape_tensor):
            return None
        return (...,) + tuple(
            slice(int(slice_start), int(slice_end), int(step_size))
            for (slice_start, slice_end, step_size) in zip(
                translation, slice_ends, scale
            )
        )


class VoxelCoordinateGrid(BaseVoxelCoordinateGrid):
    """Voxel coordinate grid (possibly transformed)"""

    def __init__(
        self,
        shape: Sequence[int],
        affine_transformation: Optional[IAffineTransformation] = None,
    ) -> None:
        self._shape = shape
        self._affine_transformation: IAffineTransformation = (
            Identity(len(self._shape))
            if affine_transformation is None
            else affine_transformation
        )

    @property
    def mask(self) -> Optional[Tensor]:
        return None

    def generate_mask(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None:
            raise RuntimeError("Device is needed!")
        return ones(
            reduce_channel_shape_to_ones((1, 1) + tuple(self._shape), 1),
            device=device,
            dtype=dtype,
        )

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None:
            raise RuntimeError("Device is needed!")
        return self._affine_transformation(
            generate_voxel_coordinate_grid(self._shape, device, dtype=dtype)
        )

    def apply_affine(
        self, affine_transformation: IAffineTransformation
    ) -> "VoxelCoordinateGrid":
        return self.modified_copy(
            affine_transformation=affine_transformation.compose_affine(
                self._affine_transformation
            )
        )

    @property
    def channels_shape(self) -> Sequence[int]:
        return (len(self.shape),)

    def has_mask(self) -> bool:
        return False

    def modified_copy(
        self,
        shape: Optional[Sequence[int]] = None,
        affine_transformation: Optional[IAffineTransformation] = None,
    ) -> "VoxelCoordinateGrid":
        """Create modified copy"""
        return VoxelCoordinateGrid(
            self._shape if shape is None else shape,
            self._affine_transformation
            if affine_transformation is None
            else affine_transformation,
        )

    def get_cpu_affine(self) -> Optional[ICPUComposableAffineTransformation]:
        if isinstance(self._affine_transformation, ICPUComposableAffineTransformation):
            return self._affine_transformation
        return None

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    def clear_mask(self) -> "VoxelCoordinateGrid":
        return self

    def detach(self) -> "VoxelCoordinateGrid":
        return self


class _CachedVoxelCoordinateGrid(BaseVoxelCoordinateGrid, _CachedMaskedTensor):
    def __init__(
        self,
        masked_tensor: IMaskedTensor,
    ) -> None:
        self._masked_tensor: IRegularGridTensor
        super().__init__(masked_tensor)

    def apply_affine(
        self, affine_transformation: IAffineTransformation
    ) -> "_CachedVoxelCoordinateGrid":
        return self._create_with_existing_cache(
            masked_tensor=self._masked_tensor,
            affine_transformation=affine_transformation.compose_affine(
                self._affine_transformation
            ),
            values_cache=self._values_cache,
            mask_cache=self._mask_cache,
        )

    def clear_mask(self) -> "_CachedVoxelCoordinateGrid":
        return self._create_with_existing_cache(
            masked_tensor=self._masked_tensor,
            affine_transformation=self._affine_transformation,
            values_cache=self._values_cache,
            mask_cache=None,
        )

    def get_cpu_affine(self) -> Optional[ICPUComposableAffineTransformation]:
        if isinstance(self._affine_transformation, ICPUComposableAffineTransformation):
            other_cpu_affine = self._masked_tensor.get_cpu_affine()
            if other_cpu_affine is not None:
                composed = self._affine_transformation.compose_affine(other_cpu_affine)
                if isinstance(composed, ICPUComposableAffineTransformation):
                    return composed
        return None

    @property
    def shape(self) -> Sequence[int]:
        return self._masked_tensor.shape
