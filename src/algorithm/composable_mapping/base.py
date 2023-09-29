"""Base classes for composable mapping"""

from torch import device as torch_device
from torch import dtype as torch_dtype

from algorithm.composable_mapping.interface import IComposableMapping

from .interface import IComposableMapping, IMaskedTensor


class BaseComposableMapping(IComposableMapping):
    """Base class for composable mappings"""

    def compose(self, mapping: "IComposableMapping") -> "IComposableMapping":
        return _Composition(self, mapping)


class _Composition(BaseComposableMapping):
    """Composition of two mappings"""

    def __init__(self, left_mapping: IComposableMapping, right_mapping: IComposableMapping) -> None:
        self._left_mapping = left_mapping
        self._right_mapping = right_mapping

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self._left_mapping(self._right_mapping(masked_coordinates))

    def invert(self, **inversion_parameters) -> "IComposableMapping":
        return _Composition(
            self._right_mapping.invert(**inversion_parameters),
            self._left_mapping.invert(**inversion_parameters),
        )

    def detach(self) -> "_Composition":
        return _Composition(self._left_mapping.detach(), self._right_mapping.detach())

    def __repr__(self) -> str:
        return (
            f"<algorithm.composable_mapping.base._Composition, "
            f"left: {self._left_mapping}, right: {self._right_mapping}>"
        )

    def to_dtype(self, dtype: torch_dtype) -> "_Composition":
        return _Composition(self._left_mapping.to_dtype(dtype), self._right_mapping.to_dtype(dtype))

    def to_device(self, device: torch_device) -> "_Composition":
        return _Composition(
            self._left_mapping.to_device(device), self._right_mapping.to_device(device)
        )
