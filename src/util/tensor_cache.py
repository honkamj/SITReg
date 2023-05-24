"""Simple cache for caching tensors based on device"""

from typing import Callable, Optional

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype


class TensorCache:
    """Simple cache for caching tensors based on device"""

    def __init__(self, obtainer: Callable[[torch_device, torch_dtype], Tensor]) -> None:
        self._tensors: dict[tuple[torch_device, torch_dtype], Optional[Tensor]] = {}
        self._obtainer = obtainer

    @classmethod
    def _create_with_tensors(
        cls,
        obtainer: Callable[[torch_device, torch_dtype], Tensor],
        tensors: dict[tuple[torch_device, torch_dtype], Optional[Tensor]],
    ) -> "TensorCache":
        tensor_cache = cls(obtainer)
        tensor_cache._tensors = tensors
        return tensor_cache

    def get(self, device: torch_device, dtype: torch_dtype) -> Tensor:
        """Get value"""
        value = self._tensors.get((device, dtype))
        if value is None:
            value = self._obtainer(device, dtype)
            self._tensors[(device, dtype)] = value
        return value

    def detach(self) -> "TensorCache":
        """Detach the cached objects from computational graph"""
        detached_tensors: dict[tuple[torch_device, torch_dtype], Optional[Tensor]] = {}
        for key, tensor in self._tensors.items():
            detached_tensors[key] = None if tensor is None else tensor.detach()
        return self._create_with_tensors(obtainer=self._obtainer, tensors=detached_tensors)
