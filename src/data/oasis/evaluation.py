"""OASIS dataset evaluation"""

from typing import Mapping, Sequence

from data.base import BaseVolumetricRegistrationSegmentationEvaluator


class OasisEvaluator(BaseVolumetricRegistrationSegmentationEvaluator):
    """Oasis evaluator"""

    NAMES_TO_INDICES_SEG35: dict[str, Sequence[int]] = {
        "Left-Cerebral-White-Matter": [1],
        "Right-Cerebral-White-Matter": [20],
        "Left-Cerebral-Cortex": [2],
        "Right-Cerebral-Cortex": [21],
        "Left-Lateral-Ventricle": [3],
        "Right-Lateral-Ventricle": [22],
        "Left-Inf-Lat-Ventricle": [4],
        "Right-Inf-Lat-Ventricle": [23],
        "Left-Cerebellum-White-Matter": [5],
        "Right-Cerebellum-White-Matter": [24],
        "Left-Cerebellum-Cortex": [6],
        "Right-Cerebellum-Cortex": [25],
        "Left-Thalamus": [7],
        "Right-Thalamus": [26],
        "Left-Caudate": [8],
        "Right-Caudate": [27],
        "Left-Putamen": [9],
        "Right-Putamen": [28],
        "Left-Palladum": [10],
        "Right-Palladum": [29],
        "3rd-Ventricle": [11],
        "4th-Ventricle": [12],
        "Brain-Stem": [13],
        "Left-Hippocampus": [14],
        "Right-Hippocampus": [30],
        "Left-Amygdala": [15],
        "Right-Amygdala": [31],
        "Left-Accumbens": [16],
        "Right-Accumbens": [32],
        "Left-Ventral-DC": [17],
        "Right-Ventral-DC": [33],
        "Left-Vessel": [18],
        "Right-Vessel": [34],
        "Left-Choroid-Plexus": [19],
        "Right-Choroid-Plexus": [35],
    }

    @property
    def _names_to_indices_seg(self) -> Mapping[str, Sequence[int]]:
        return self.NAMES_TO_INDICES_SEG35
