"""Affine transformation implementations"""


from typing import Optional, Sequence

from torch import Tensor, allclose
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import eye, inverse, matmul
from algorithm.composable_mapping.interface import IComposableMapping

from util.dimension_order import (
    broadcast_tensors_by_leading_dims,
    channels_last,
    merged_batch_dimensions,
)
from util.tensor_cache import TensorCache

from ..affine_transformation import (
    compose_affine_transformation_matrices,
    convert_to_homogenous_coordinates,
)
from .base import BaseComposableMapping
from .interface import (
    IAffineTransformation,
    ICPUComposableAffineTransformation,
    IComposableMapping,
    IMaskedTensor,
)


@channels_last({"coordinates": 1, "transformation_matrix": 2}, 1)
@merged_batch_dimensions({"coordinates": 1, "transformation_matrix": 2}, 1)
def _transform_coordinates(coordinates: Tensor, transformation_matrix: Tensor) -> Tensor:
    transformed = matmul(
        transformation_matrix, convert_to_homogenous_coordinates(coordinates)[..., None]
    )[..., :-1, 0]
    return transformed


def _broadcast_and_transform_coordinates(
    coordinates: Tensor, transformation_matrix: Tensor
) -> Tensor:
    coordinates, transformation_matrix = broadcast_tensors_by_leading_dims(
        (coordinates, transformation_matrix), num_leading_dims=(1, 2)
    )
    return _transform_coordinates(coordinates, transformation_matrix)


class BaseAffine(IAffineTransformation):
    """Base affine transformation"""

    def compose_affine(
        self, affine_transformation: IAffineTransformation
    ) -> "IAffineTransformation":
        return compose_affine_transformations(self, affine_transformation)


class Affine(BaseAffine):
    """Represents generic affine transformation

    Arguments:
        transformation_matrix: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, ...),
            if None, corresponds to identity transformation
    """

    def __init__(self, transformation_matrix: Tensor) -> None:
        self._transformation_matrix = transformation_matrix

    def __call__(self, coordinates: Tensor) -> Tensor:
        return _broadcast_and_transform_coordinates(coordinates, self._transformation_matrix)

    def invert(self) -> "Affine":
        """Invert the transformation"""
        return Affine(channels_last(2, 2)(inverse)(self._transformation_matrix))

    def as_matrix(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        return self._transformation_matrix

    @property
    def device(self) -> torch_device:
        """Device of the transformation matrix"""
        return self._transformation_matrix.device

    @property
    def dtype(self) -> torch_dtype:
        """Dtype of the transformation matrix"""
        return self._transformation_matrix.dtype

    def detach(self) -> "Affine":
        return Affine(self._transformation_matrix.detach())

    def to_device(self, device: torch_device) -> "Affine":
        return Affine(self._transformation_matrix.to(device=device))

    def to_dtype(self, dtype: torch_dtype) -> "Affine":
        return Affine(self._transformation_matrix.type(dtype))


class Identity(BaseAffine, ICPUComposableAffineTransformation):
    """Identity transformation"""

    def __init__(self, n_dims: int) -> None:
        self._n_dims = n_dims

    def __call__(self, coordinates: Tensor) -> Tensor:
        return coordinates

    def invert(self) -> "Identity":
        """Invert the transformation"""
        return Identity(self._n_dims)

    def as_matrix(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None:
            raise RuntimeError("Device is needed!")
        return eye(self._n_dims + 1, device=device, dtype=dtype)

    def as_cpu_matrix(self, dtype: Optional[torch_dtype] = None) -> Tensor:
        return eye(self._n_dims + 1, device=torch_device("cpu"), dtype=dtype)

    def detach(self) -> "Identity":
        return self

    def to_device(self, device: torch_device) -> "Identity":
        return self

    def to_dtype(self, dtype: torch_dtype) -> "Identity":
        return self


class CPUComposableAffine(BaseAffine, ICPUComposableAffineTransformation):
    """Base class for affine tranformations for which compositions and
    inversions are done actively on CPU, and PyTorch tensors on target devices
    are created on request

    Allows to do decisions on CPU on whether to perform some more costly
    computation on GPU. Usually one should create the cached version
    by calling cache method after initialization. Otherwise the gpu
    version will have to be recreated each time.

    Arguments:
        transformation_matrix_on_cpu: Transformation matrix on cpu
    """

    def __init__(self, transformation_matrix_on_cpu: Tensor) -> None:
        if transformation_matrix_on_cpu.device != torch_device("cpu"):
            raise ValueError("Please give the matrix on CPU!")
        self._transformation_matrix_cpu = transformation_matrix_on_cpu.detach().squeeze()

    def __call__(self, coordinates: Tensor) -> Tensor:
        if self._is_identity():
            return coordinates
        return _broadcast_and_transform_coordinates(coordinates, self.as_matrix(coordinates.device))

    def as_matrix(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        if device is None:
            device = torch_device("cpu")
        return self._as_matrix(device)

    def _as_matrix(self, device: torch_device) -> Tensor:
        return self._transformation_matrix_cpu.to(device)

    def as_cpu_matrix(self, dtype: Optional[torch_dtype] = None) -> Tensor:
        """Return transformation matrix on cpu"""
        return self._transformation_matrix_cpu

    def _is_identity(self) -> bool:
        if self._transformation_matrix_cpu.ndim == 2:
            n_dims = self._transformation_matrix_cpu.size(0)
            return allclose(
                self._transformation_matrix_cpu,
                eye(n_dims, dtype=self._transformation_matrix_cpu.dtype),
            )
        return False

    def invert(self) -> "_CPUComposableAffineInverse":
        return _CPUComposableAffineInverse(self)

    def cache(self) -> "_CPUComposableAffineCache":
        """Generates version which caches the computed Tensors"""
        return _CPUComposableAffineCache(self)

    def compose_cpu_affine(
        self, affine_transformation: "CPUComposableAffine"
    ) -> "CPUComposableAffine":
        """Compose lazy affine transformation"""
        return _CPUComposableAffineComposition(self, affine_transformation)

    def detach(self) -> "CPUComposableAffine":
        return self

    def to_dtype(self, dtype: torch_dtype) -> "CPUComposableAffine":
        return CPUComposableAffine(self._transformation_matrix_cpu.type(dtype))

    def to_device(self, device: torch_device) -> "CPUComposableAffine":
        return self


class _CPUComposableAffineComposition(CPUComposableAffine):
    def __init__(
        self, left_transformation: CPUComposableAffine, right_transformation: CPUComposableAffine
    ) -> None:
        self._left_transformation = left_transformation
        self._right_transformation = right_transformation
        super().__init__(
            compose_affine_transformation_matrices(
                left_transformation.as_cpu_matrix(), right_transformation.as_cpu_matrix()
            )
        )

    def _as_matrix(self, device: torch_device) -> Tensor:
        return compose_affine_transformation_matrices(
            self._left_transformation.as_matrix(device),
            self._right_transformation.as_matrix(device),
        )

    def to_dtype(self, dtype: torch_dtype) -> "_CPUComposableAffineComposition":
        return _CPUComposableAffineComposition(
            self._left_transformation.to_dtype(dtype), self._right_transformation.to_dtype(dtype)
        )

    def to_device(self, device: torch_device) -> "_CPUComposableAffineComposition":
        return self

    def detach(self) -> "_CPUComposableAffineComposition":
        return self


class _CPUComposableAffineInverse(CPUComposableAffine):
    def __init__(self, transformation_to_invert: CPUComposableAffine) -> None:
        self._transformation_to_invert = transformation_to_invert
        super().__init__(channels_last(2, 2)(inverse)(transformation_to_invert.as_cpu_matrix()))

    def _as_matrix(self, device: torch_device) -> Tensor:
        return channels_last(2, 2)(inverse)(self._transformation_to_invert.as_matrix(device))

    def to_dtype(self, dtype: torch_dtype) -> "_CPUComposableAffineInverse":
        return _CPUComposableAffineInverse(
            self._transformation_to_invert.to_dtype(dtype)
        )

    def to_device(self, device: torch_device) -> "_CPUComposableAffineInverse":
        return self

    def detach(self) -> "_CPUComposableAffineInverse":
        return self


class _CPUComposableAffineCache(CPUComposableAffine):
    def __init__(self, transformation_to_cache: CPUComposableAffine) -> None:
        self._transformation_to_cache = transformation_to_cache
        self._matrix_cache = TensorCache(transformation_to_cache.as_matrix)
        super().__init__(transformation_to_cache.as_cpu_matrix())

    def _as_matrix(self, device: torch_device) -> Tensor:
        return self._matrix_cache.get(device=device, dtype=self._transformation_matrix_cpu.dtype)

    def detach(self) -> "_CPUComposableAffineCache":
        return self

    def to_dtype(self, dtype: torch_dtype) -> "_CPUComposableAffineCache":
        return _CPUComposableAffineCache(self._transformation_to_cache.to_dtype(dtype))

    def to_device(self, device: torch_device) -> "_CPUComposableAffineCache":
        return self


def compose_affine_transformations(
    left_affine: IAffineTransformation, right_affine: IAffineTransformation
) -> "IAffineTransformation":
    """Compose two affine transformations"""
    if isinstance(left_affine, Identity):
        return right_affine
    if isinstance(right_affine, Identity):
        return left_affine
    if isinstance(left_affine, CPUComposableAffine) and isinstance(
        right_affine, CPUComposableAffine
    ):
        return left_affine.compose_cpu_affine(right_affine)
    device: Optional[torch_device] = None
    dtype: Optional[torch_dtype] = None
    if isinstance(left_affine, Affine):
        device = left_affine.device
        dtype = left_affine.dtype
    if isinstance(right_affine, Affine):
        device = right_affine.device
        dtype = right_affine.dtype
    return Affine(
        compose_affine_transformation_matrices(
            left_affine.as_matrix(device=device, dtype=dtype),
            right_affine.as_matrix(device=device, dtype=dtype),
        )
    )


class ComposableAffine(BaseComposableMapping):
    """Composable wrapper for affine transformations"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self._affine_transformation = affine_transformation

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates.apply_affine(self._affine_transformation)

    def invert(self, **_inversion_parameters) -> "ComposableAffine":
        return ComposableAffine(self._affine_transformation.invert())

    def detach(self) -> "ComposableAffine":
        return ComposableAffine(self._affine_transformation.detach())

    def to_device(self, device: torch_device) -> "ComposableAffine":
        return ComposableAffine(self._affine_transformation.to_device(device))

    def to_dtype(self, dtype: torch_dtype) -> "ComposableAffine":
        return ComposableAffine(self._affine_transformation.to_dtype(dtype))


def as_affine_transformation(
    composable_mapping: IComposableMapping, n_dims: int
) -> IAffineTransformation:
    """Extract affine mapping from composable mapping

    Raises an error if the composable mapping is not fully affine.
    """
    tracer = _AffineTracer(Identity(n_dims))
    traced = composable_mapping(tracer)
    if isinstance(traced, _AffineTracer):
        return traced.affine_transformation
    raise RuntimeError("Could not infer affine transformation")


class _AffineTracer(IMaskedTensor):
    """Can be used to trace affine component of a composable mapping"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self.affine_transformation = affine_transformation

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        raise RuntimeError(
            "Affine tracer has no values! Usually this error means that "
            "the traced mapping is not affine."
        )

    @property
    def mask(self) -> Optional[Tensor]:
        return None

    def generate_mask(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        raise RuntimeError(
            "Affine tracer has no mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        return _AffineTracer(affine_transformation.compose_affine(self.affine_transformation))

    def has_mask(self) -> bool:
        return False

    @property
    def channels_shape(self) -> Sequence[int]:
        raise RuntimeError(
            "Affine tracer has no channels! Usually this error means that "
            "the traced mapping is not affine."
        )

    @property
    def shape(self) -> Sequence[int]:
        raise RuntimeError(
            "Affine tracer has no shape! Usually this error means that "
            "the traced mapping is not affine."
        )

    def cache(self) -> "_AffineTracer":
        return self

    def clear_mask(self) -> "_AffineTracer":
        return self

    def detach(self) -> "_AffineTracer":
        return self
