"""Loss related utils"""

from typing import Any, Mapping


def handle_params(
    params: Mapping[str, Any] | None = None,
    default_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Handle params dictionary for losses"""
    if params is None:
        params = {}
    if default_params is None:
        default_params = {}
    return dict(default_params) | dict(params)


def build_default_params(
    none_allowed_params: Mapping[str, Any] | None = None,
    none_ignored_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build default params dictionary from values which are not None"""
    default_params = {}
    if none_allowed_params is not None:
        for key, value in none_allowed_params.items():
            default_params[key] = value
    if none_ignored_params is not None:
        for key, value in none_ignored_params.items():
            if value is not None:
                default_params[key] = value
    return default_params
