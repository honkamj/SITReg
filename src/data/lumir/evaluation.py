"""OASIS dataset evaluation"""

from typing import Mapping, Sequence

from data.base import BaseVolumetricRegistrationSegmentationEvaluator


class LumirEvaluator(BaseVolumetricRegistrationSegmentationEvaluator):
    """LUMIR evaluator"""

    NAMES_TO_INDICES_SEG: dict[str, Sequence[int]] = {
        "label_0": [0],
        "label_4": [4],
        "label_11": [11],
        "label_23": [23],
        "label_30": [30],
        "label_31": [31],
        "label_32": [32],
        "label_35": [35],
        "label_36": [36],
        "label_37": [37],
        "label_38": [38],
        "label_39": [39],
        "label_40": [40],
        "label_41": [41],
        "label_44": [44],
        "label_45": [45],
        "label_47": [47],
        "label_48": [48],
        "label_49": [49],
        "label_50": [50],
        "label_51": [51],
        "label_52": [52],
        "label_55": [55],
        "label_56": [56],
        "label_57": [57],
        "label_58": [58],
        "label_59": [59],
        "label_60": [60],
        "label_61": [61],
        "label_62": [62],
        "label_71": [71],
        "label_72": [72],
        "label_73": [73],
        "label_75": [75],
        "label_76": [76],
        "label_100": [100],
        "label_101": [101],
        "label_102": [102],
        "label_103": [103],
        "label_104": [104],
        "label_105": [105],
        "label_106": [106],
        "label_107": [107],
        "label_108": [108],
        "label_109": [109],
        "label_112": [112],
        "label_113": [113],
        "label_114": [114],
        "label_115": [115],
        "label_116": [116],
        "label_117": [117],
        "label_118": [118],
        "label_119": [119],
        "label_120": [120],
        "label_121": [121],
        "label_122": [122],
        "label_123": [123],
        "label_124": [124],
        "label_125": [125],
        "label_128": [128],
        "label_129": [129],
        "label_132": [132],
        "label_133": [133],
        "label_134": [134],
        "label_135": [135],
        "label_136": [136],
        "label_137": [137],
        "label_138": [138],
        "label_139": [139],
        "label_140": [140],
        "label_141": [141],
        "label_142": [142],
        "label_143": [143],
        "label_144": [144],
        "label_145": [145],
        "label_146": [146],
        "label_147": [147],
        "label_148": [148],
        "label_149": [149],
        "label_150": [150],
        "label_151": [151],
        "label_152": [152],
        "label_153": [153],
        "label_154": [154],
        "label_155": [155],
        "label_156": [156],
        "label_157": [157],
        "label_160": [160],
        "label_161": [161],
        "label_162": [162],
        "label_163": [163],
        "label_164": [164],
        "label_165": [165],
        "label_166": [166],
        "label_167": [167],
        "label_168": [168],
        "label_169": [169],
        "label_170": [170],
        "label_171": [171],
        "label_172": [172],
        "label_173": [173],
        "label_174": [174],
        "label_175": [175],
        "label_176": [176],
        "label_177": [177],
        "label_178": [178],
        "label_179": [179],
        "label_180": [180],
        "label_181": [181],
        "label_182": [182],
        "label_183": [183],
        "label_184": [184],
        "label_185": [185],
        "label_186": [186],
        "label_187": [187],
        "label_190": [190],
        "label_191": [191],
        "label_192": [192],
        "label_193": [193],
        "label_194": [194],
        "label_195": [195],
        "label_196": [196],
        "label_197": [197],
        "label_198": [198],
        "label_199": [199],
        "label_200": [200],
        "label_201": [201],
        "label_202": [202],
        "label_203": [203],
        "label_204": [204],
        "label_205": [205],
        "label_206": [206],
        "label_207": [207],
    }

    @property
    def _names_to_indices_seg(self) -> Mapping[str, Sequence[int]]:
        return self.NAMES_TO_INDICES_SEG
