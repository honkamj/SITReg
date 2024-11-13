"""Segmentation losses"""

from typing import Any, Mapping

from composable_mapping import MappableTensor
from torch import Tensor

from util.dimension_order import get_other_than_batch_dim

from .interface import ISegmentationLoss
from .util import build_default_params, handle_params


class DiceLoss(ISegmentationLoss):
    """Dice loss"""

    def __init__(self, smooth: float | None = 1e-5) -> None:
        self._default_params = build_default_params(
            none_ignored_params={"smooth": smooth, "weight": 1.0}
        )

    def __call__(
        self,
        seg_1: MappableTensor,
        seg_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        params = handle_params(params, self._default_params)
        masked_seg_1 = self._as_masked(seg_1)
        masked_seg_2 = self._as_masked(seg_2)
        spatial_dims = list(range(2, masked_seg_1.ndim))
        intersection = (masked_seg_1 * masked_seg_2).sum(dim=spatial_dims)
        union = masked_seg_1.sum(dim=spatial_dims) + masked_seg_2.sum(dim=spatial_dims)

        dice = (2 * intersection + params["smooth"]) / (union + params["smooth"])

        return (-dice.mean(dim=get_other_than_batch_dim(dice)) * params["weight"]).mean()

    @staticmethod
    def _as_masked(
        seg: MappableTensor,
    ) -> Tensor:
        values, mask = seg.generate(generate_missing_mask=True, cast_mask=True)
        return values * mask
