"""Type casting and conversion utilities with type checking"""

from typing import Any, Iterable, TypeVar, cast


T = TypeVar("T")


def cast_with_check(item_type: type[T], item: Any) -> T:
    """Cast to type with type checking"""
    if not isinstance(item, item_type):
        raise ValueError(f"Invalid type, expected {item_type}, got {type(item)}") 
    return item


def cast_to_iterable_with_check(item: Any) -> Iterable:
    """Cast to type with type checking"""
    if not isinstance(item, Iterable):
        raise ValueError(f"Invalid type, expected Iterable, got {type(item)}")
    return item


def to_two_tuple(elem_type: type[T], item: Any) -> tuple[T, T]:
    """Convert to tuple of length 2"""
    output = tuple(cast_to_iterable_with_check(item))
    if len(output) != 2:
        raise ValueError(f"Invalid length, expected 2, got {len(output)}")
    if not isinstance(output[0], elem_type):
        raise ValueError(
            "Invalid element type at position 0, "
            f"expected {elem_type}, got {type(output[0])}"
        )
    if not isinstance(output[1], elem_type):
        raise ValueError(
            "Invalid element type at position 1, "
            f"expected {elem_type}, got {type(output[1])}"
        )
    return cast(tuple[T, T], output)


def to_optional_two_tuple(elem_type: type[T], item: Any) -> tuple[T, T] | None:
    """Convert optionally to tuple of length 2"""
    if item is None:
        return None
    return to_two_tuple(elem_type, item)


def to_list_of_two_tuples(elem_type: type[T], item: Any) -> list[tuple[T, T]]:
    """Convert to list of tuples of length 2"""
    item_list = []
    for two_tuple_candidate in cast_to_iterable_with_check(item):
        item_list.append(to_two_tuple(elem_type, two_tuple_candidate))
    return item_list


def to_optional_list_of_two_tuples(elem_type: type[T], item: Any) -> list[tuple[T, T]] | None:
    """Convert optionally to list of tuples of length 2"""
    if item is None:
        return None
    return to_list_of_two_tuples(elem_type, item)


def to_tuple(elem_type: type[T], item: Any) -> tuple[T, ...]:
    """Convert to tuple of any length"""
    iterable_item = cast_to_iterable_with_check(item)
    if not all(isinstance(elem, elem_type) for elem in iterable_item):
        raise ValueError(f"Invalid element type, expected {elem_type}")
    return tuple(iterable_item)


def to_optional_tuple(elem_type: type[T], item: Any) -> tuple[T, ...] | None:
    """Convert optionally to tuple with any length"""
    if item is None:
        return None
    return to_tuple(elem_type, item)


def to_list_of_optional_tuples(elem_type: type[T], item: Any) -> list[tuple[T, ...] | None]:
    """Convert to list of optional tuples of any length"""
    item_list = []
    for tuple_candidate in cast_to_iterable_with_check(item):
        item_list.append(to_optional_tuple(elem_type, tuple_candidate))
    return item_list
