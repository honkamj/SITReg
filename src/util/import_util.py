"""Utility functions for importing"""

from importlib import import_module
from typing import Any


def import_object(import_path: str) -> Any:
    """Load object from the defined import path"""
    split = import_path.split(".")
    module_path = ".".join(split[:-1])
    object_name = split[-1]
    imported_object = getattr(import_module(module_path), object_name)
    return imported_object
