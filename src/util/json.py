"""JSON related utils"""

from json import dump as json_dump
from json import load as json_load
from typing import Any


def save_json(path: str, data: Any) -> None:
    """Save json to path"""
    with open(
        path,
        mode="w",
        encoding="utf-8",
    ) as case_metrics_file:
        json_dump(data, case_metrics_file, indent=4)


def load_json(path: str) -> dict[str, Any]:
    """Load json from path"""
    with open(
        path,
        mode="r",
        encoding="utf-8",
    ) as case_metrics_file:
        return json_load(case_metrics_file)
