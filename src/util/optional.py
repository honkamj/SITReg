"""Util for handling optional values"""


from typing import Optional, TypeVar, cast

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
