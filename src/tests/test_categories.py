"""Test decorators for marking some tests as lengthy"""

from functools import wraps
from os import environ
from unittest import TestCase


def test_wrapper(category: str):
    """Wraps test such that the test is ran only if the given category is specified
    in environment varible TEST_CATEGORIES_TO_RUN"""

    def _wrapper_func(func):
        @wraps(func)
        def _modified_func(test_case: TestCase, *args, **kwargs):
            categories = [
                available_category.lower()
                for available_category in environ.get("TEST_CATEGORIES_TO_RUN", "").split(",")
            ]
            if category.lower() in categories:
                return func(test_case, *args, **kwargs)
            test_case.skipTest(
                reason=(
                    f'Test category "{category}" is not included in the environment '
                    "variable TEST_CATEGORIES_TO_RUN."
                )
            )

        return _modified_func

    return _wrapper_func


lengthy = test_wrapper("lengthy")
