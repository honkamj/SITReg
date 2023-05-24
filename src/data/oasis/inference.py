"""Inference data related implementations"""

from typing import Any, Mapping

from torch import Tensor
from torch import device as torch_device

from data.dataset import VolumetricRegistrationInferenceDataset
from data.oasis.evaluation import OasisEvaluator
from data.storage import NiftiStorageFactory

from ..base import BaseVolumetricRegistrationInferenceFactory


class OasisInferenceFactory(BaseVolumetricRegistrationInferenceFactory):
    """Oasis inference factory"""

    def __init__(
        self, dataset: VolumetricRegistrationInferenceDataset, data_config: Mapping[str, Any]
    ) -> None:
        super().__init__(dataset)
        self._evaluation_mask_file_type_seg35 = data_config["evaluation_mask_file_type_seg35"]
        self._metrics_to_compute = data_config["metrics"][dataset.division]

    def _get_storage_factory(self, affine: Tensor) -> NiftiStorageFactory:
        return NiftiStorageFactory(affine)

    def get_evaluator(self, index: int, device: torch_device) -> OasisEvaluator:
        source_mask_seg35, target_mask_seg35 = self._dataset.get_pair(
            index,
            data_args=self._dataset.data_args.modified_copy(
                file_type=self._evaluation_mask_file_type_seg35,
                normalize=False,
                shift_and_normalize=None,
            ),
        )
        image_1_name, image_2_name = self._dataset.names(index)
        image_1_affine, image_2_affine = self._dataset.affines(index)
        return OasisEvaluator(
            source_mask_seg=source_mask_seg35.to(device),
            target_mask_seg=target_mask_seg35.to(device),
            metrics_to_compute=self._metrics_to_compute,
            source_temp_storage_factory=self._get_storage_factory(image_1_affine),
            source_name=image_1_name,
            target_temp_storage_factory=self._get_storage_factory(image_2_affine),
            target_name=image_2_name
        )
