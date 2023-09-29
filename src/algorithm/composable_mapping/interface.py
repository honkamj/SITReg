"""Interfaces for composable mapping"""

from abc import ABC, abstractmethod
from types import EllipsisType
from typing import Optional, Sequence

from attr import define
from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype


class IAffineTransformation(ABC):
    """Affine mapping"""

    @abstractmethod
    def compose_affine(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation":
        """Compose with affine mapping"""

    @abstractmethod
    def as_matrix(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        """Return the mapping as matrix

        Specifying a device or dtype does not guarantee that the returned Tensor
        has that dtype or device. The arguments are used only when there is no
        wrapped underlying Tensor to infer them from.
        """

    @abstractmethod
    def __call__(self, coordinates: Tensor) -> Tensor:
        """Evaluate the mapping at coordinates"""

    @abstractmethod
    def invert(self) -> "IAffineTransformation":
        """Invert the transformation"""

    @abstractmethod
    def detach(self) -> "IAffineTransformation":
        """Detach the underlying Tensors from computational graph

        Might return self.
        """

    @abstractmethod
    def to_device(self, device: torch_device) -> "IAffineTransformation":
        """Put underlying Tensors to given device"""

    @abstractmethod
    def to_dtype(self, dtype: torch_dtype) -> "IAffineTransformation":
        """Put underlying Tensors to given dtype"""


class ICPUComposableAffineTransformation(IAffineTransformation):
    """Affine mapping which has separate representation on CPU

    Allows doing decision or simplifications on CPU and hence
    preventing or simplifying more costly actions on GPU.
    """

    @abstractmethod
    def as_cpu_matrix(self, dtype: Optional[torch_dtype] = None) -> Tensor:
        """Returns the mapping on cpu

        Specifying a dtype does not guarantee that the returned Tensor has that
        dtype. The argument is used only when there is no wrapped underlying
        Tensor to infer dtype from.
        """


class IMaskedTensor(ABC):
    """Wrapper for masked tensor"""

    @abstractmethod
    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        """Obtain values contained by the class"""

    @property
    @abstractmethod
    def mask(self) -> Optional[Tensor]:
        """Obtain mask contained by the class

        Number of dimensions matches the values except
        that channel dimensions are changed to size 1.
        """

    @abstractmethod
    def generate_mask(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        """Generate mask with correct shape even if not defined

        Specifying a device or dtype does not guarantee that the returned Tensor
        has that dtype or device. The arguments are used only when there is no
        wrapped underlying Tensor to infer them from.
        """

    @abstractmethod
    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        """Apply affine mapping to the first channel dimension of tensor"""

    @abstractmethod
    def has_mask(self) -> bool:
        """Returns whether the tensor has a mask"""

    @property
    @abstractmethod
    def channels_shape(self) -> Sequence[int]:
        """Return shape of the channels dimension"""

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the values"""

    @abstractmethod
    def cache(self) -> "IMaskedTensor":
        """Return cached version of the same masked tensor"""

    @abstractmethod
    def clear_mask(self) -> "IMaskedTensor":
        """Return version of the tensor with mask cleared"""

    @abstractmethod
    def detach(self) -> "IMaskedTensor":
        """Detach the underlying Tensors from computational graph

        Might return self.
        """


class IRegularGridTensor(IMaskedTensor):
    """Regular grid"""

    @abstractmethod
    def get_cpu_affine(self) -> Optional[ICPUComposableAffineTransformation]:
        """Return affine transformation applied to the voxel grid if it is
        ICPUComposableAffineTransformation

        If it is not ICPUComposableAffineTransformation, return None.
        """

    @abstractmethod
    def reduce_to_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[tuple[EllipsisType | slice, ...]]:
        """Reduce the grid to slice on voxel with target shape assuming it is in voxel
        coordinates

        Should be done on CPU, If impossible, returns None"""

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the grid"""


class IComposableMapping(ABC):
    """Composable mapping"""

    @abstractmethod
    def compose(self, mapping: "IComposableMapping") -> "IComposableMapping":
        """Compose with a mapping"""

    @abstractmethod
    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        """Evaluate the mapping at coordinates"""

    @abstractmethod
    def invert(self, **inversion_parameters) -> "IComposableMapping":
        """Invert the mapping

        Args:
            inversion_parameters: Possible inversion parameters
        """

    @abstractmethod
    def detach(self) -> "IComposableMapping":
        """Detach the underlying Tensors from computational graph

        Might return self.
        """

    @abstractmethod
    def to_device(self, device: torch_device) -> "IComposableMapping":
        """Put underlying Tensors to given device"""

    @abstractmethod
    def to_dtype(self, dtype: torch_dtype) -> "IComposableMapping":
        """Put underlying Tensors to given dtype"""


@define
class VoxelCoordinateSystem:
    """Represents coordinate system between voxel and world coordinates"""

    from_voxel_coordinates: IComposableMapping
    to_voxel_coordinates: IComposableMapping
    grid: IMaskedTensor
    voxel_grid: IMaskedTensor
    grid_spacing: Sequence[float]
