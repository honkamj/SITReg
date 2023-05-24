"""Nth permutation"""

from math import factorial, perm
from typing import Optional


def nth_permutation_indices(
    n_elements: int, index: int, n_elements_per_permutation: Optional[int] = None
) -> tuple[int, ...]:
    """Equivalent to list(permutations(range(n_elements), r=n_elements_per_permutation))[index]

    Modified from https://github.com/more-itertools/more-itertools
    """
    if n_elements_per_permutation is None or n_elements_per_permutation == n_elements:
        n_elements_per_permutation, permutations = n_elements, factorial(n_elements)
    elif not 0 <= n_elements_per_permutation < n_elements:
        raise ValueError("Invalid number of elements requested")
    else:
        permutations = perm(n_elements, n_elements_per_permutation)

    if index < 0:
        index += permutations

    if not 0 <= index < permutations:
        raise IndexError("Invalid index")

    if permutations == 0:
        return tuple()

    result = [0] * n_elements_per_permutation
    numerator = (
        index * factorial(n_elements - n_elements_per_permutation)
        if n_elements_per_permutation < n_elements
        else index
    )
    for denominator in range(1, n_elements + 1):
        numerator, remainder = divmod(numerator, denominator)
        if 0 <= n_elements - denominator < n_elements_per_permutation:
            result[n_elements - denominator] = remainder
        if numerator == 0:
            break

    return tuple(result)
