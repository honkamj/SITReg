"""Util for handling optional values"""


from typing import Optional, TypeVar, cast

from torch import Tensor, maximum

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def optional_add(addable_1: Optional[T1], addable_2: Optional[T2]) -> Optional[T1 | T2]:
    """Optional add"""
    if addable_1 is None:
        return addable_2
    if addable_2 is None:
        return addable_1
    added = addable_1 + addable_2  # type: ignore
    return cast(T1 | T2, added)


def optional_maximum(tensor_1: Optional[Tensor], tensor_2: Optional[Tensor]) -> Optional[Tensor]:
    """Optional maximum"""
    if tensor_1 is None:
        return tensor_2
    if tensor_2 is None:
        return tensor_1
    return maximum(tensor_1, tensor_2)
