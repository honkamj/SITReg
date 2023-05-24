"""Mask modifiers"""

from typing import Optional, Sequence

from torch import device as torch_device
from torch import dtype as torch_dtype

from algorithm.composable_mapping.base import BaseComposableMapping
from algorithm.composable_mapping.interface import IMaskedTensor
from algorithm.composable_mapping.masked_tensor import MaskedTensor
from algorithm.dense_deformation import calculate_mask_based_on_bounds


class RectangleMask(BaseComposableMapping):
    """Add values to mask based on bounds"""

    def __init__(
        self,
        min_values: Sequence[float],
        max_values: Sequence[float],
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> None:
        self._min_values = min_values
        self._max_values = max_values
        self._device = device
        self._dtype = dtype

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        coordinates = masked_coordinates.generate_values(device=self._device, dtype=self._dtype)
        updated_mask = calculate_mask_based_on_bounds(
            coordinates=masked_coordinates.generate_values(self._device),
            mask_to_update=masked_coordinates.mask,
            min_values=self._min_values,
            max_values=self._max_values,
            dtype=coordinates.dtype,
        )
        return MaskedTensor(values=coordinates, mask=updated_mask)

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Rectangle mask is not invertible")

    def detach(self) -> "RectangleMask":
        return self


class ClearMask(BaseComposableMapping):
    """Add values to mask based on bounds"""

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates.clear_mask()

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Mask clearing is not invertible")

    def detach(self) -> "ClearMask":
        return self
