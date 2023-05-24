"""Interpolation functions"""

from torch import Tensor

from .dense_deformation import interpolate, spline_interpolate
from .interface import IInterpolator


class LinearInterpolator(IInterpolator):
    """Linear interpolation"""

    def __init__(self, padding_mode: str = "border") -> None:
        self._padding_mode = padding_mode

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return interpolate(
            volume=volume, grid=coordinates, mode="bilinear", padding_mode=self._padding_mode
        )


class NearestInterpolator(IInterpolator):
    """Nearest interpolation"""

    def __init__(self, padding_mode: str = "border") -> None:
        self._padding_mode = padding_mode

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return interpolate(
            volume=volume, grid=coordinates, mode="nearest", padding_mode=self._padding_mode
        )


class SplineInterpolator(IInterpolator):
    """Spline interpolation"""

    def __init__(
        self, bound: int, degree: int = 3, extrapolate: bool = True, prefilter: bool = True
    ) -> None:
        self._degree = degree
        self._bound = bound
        self._extrapolate = extrapolate
        self._prefilter = prefilter

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return spline_interpolate(
            volume=volume,
            grid=coordinates,
            bound=self._bound,
            prefilter=self._prefilter,
            degree=self._degree,
            extrapolate=self._extrapolate,
        )
