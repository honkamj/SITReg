"""OASIS dataset evaluation"""

from typing import Mapping, Sequence

from data.base import BaseVolumetricRegistrationSegmentationEvaluator


class LPBA40Evaluator(BaseVolumetricRegistrationSegmentationEvaluator):
    """LPBA40 evaluator"""

    NAMES_TO_INDICES_SEG: dict[str, Sequence[int]] = {
        "L superior frontal gyrus": [21],
        "R superior frontal gyrus": [22],
        "L middle frontal gyrus": [23],
        "R middle frontal gyrus": [24],
        "L inferior frontal gyrus": [25],
        "R inferior frontal gyrus": [26],
        "L precentral gyrus": [27],
        "R precentral gyrus": [28],
        "L middle orbitofrontal gyrus": [29],
        "R middle orbitofrontal gyrus": [30],
        "L lateral orbitofrontal gyrus": [31],
        "R lateral orbitofrontal gyrus": [32],
        "L gyrus rectus": [33],
        "R gyrus rectus": [34],
        "L postcentral gyrus": [41],
        "R postcentral gyrus": [42],
        "L superior parietal gyrus": [43],
        "R superior parietal gyrus": [44],
        "L supramarginal gyrus": [45],
        "R supramarginal gyrus": [46],
        "L angular gyrus": [47],
        "R angular gyrus": [48],
        "L precuneus": [49],
        "R precuneus": [50],
        "L superior occipital gyrus": [61],
        "R superior occipital gyrus": [62],
        "L middle occipital gyrus": [63],
        "R middle occipital gyrus": [64],
        "L inferior occipital gyrus": [65],
        "R inferior occipital gyrus": [66],
        "L cuneus": [67],
        "R cuneus": [68],
        "L superior temporal gyrus": [81],
        "R superior temporal gyrus": [82],
        "L middle temporal gyrus": [83],
        "R middle temporal gyrus": [84],
        "L inferior temporal gyrus": [85],
        "R inferior temporal gyrus": [86],
        "L parahippocampal gyrus": [87],
        "R parahippocampal gyrus": [88],
        "L lingual gyrus": [89],
        "R lingual gyrus": [90],
        "L fusiform gyrus": [91],
        "R fusiform gyrus": [92],
        "L insular cortex": [101],
        "R insular cortex": [102],
        "L cingulate gyrus": [121],
        "R cingulate gyrus": [122],
        "L caudate": [161],
        "R caudate": [162],
        "L putamen": [163],
        "R putamen": [164],
        "L hippocampus": [165],
        "R hippocampus": [166],
        "cerebellum": [181],
        "brainstem": [182],
    }

    @property
    def _names_to_indices_seg(self) -> Mapping[str, Sequence[int]]:
        return self.NAMES_TO_INDICES_SEG
