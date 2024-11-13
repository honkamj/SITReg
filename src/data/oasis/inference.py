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
        self._metrics_to_compute = data_config["metrics"][dataset.division]
        self._n_jacobian_samples = data_config.get("n_jacobian_samples_in_evaluation")
        self._evaluation_prefix = data_config.get("evaluation_prefix", "")
        self._jacobian_sampling_base_seed = data_config.get("jacobian_sampling_base_seed", None)
        self._upsampling_factor = data_config.get("downsampling_factor", None)

    def _get_storage_factory(self, affine: Tensor) -> NiftiStorageFactory:
        return NiftiStorageFactory(affine)

    def get_evaluator(self, index: int, device: torch_device) -> OasisEvaluator:
        image_1_name, image_2_name = self._dataset.names(index)
        source_mask_seg35 = self._dataset.data.get_case_evaluation_segmentation(
            case_name=image_1_name,
            args=self._dataset.data_args,
            registration_index=0,
        )[None]
        target_mask_seg35 = self._dataset.data.get_case_evaluation_segmentation(
            case_name=image_2_name, args=self._dataset.data_args, registration_index=1
        )[None]
        image_1_affine, image_2_affine = self._dataset.affines(index)
        return OasisEvaluator(
            source_mask_seg=source_mask_seg35.to(device),
            target_mask_seg=target_mask_seg35.to(device),
            metrics_to_compute=self._metrics_to_compute,
            source_temp_storage_factory=self._get_storage_factory(image_1_affine),
            source_name=image_1_name,
            target_temp_storage_factory=self._get_storage_factory(image_2_affine),
            target_name=image_2_name,
            n_jacobian_samples=self._n_jacobian_samples,
            jacobian_sampling_seed=self._jacobian_sampling_base_seed ^ index
            if self._jacobian_sampling_base_seed is not None
            else None,
            evaluation_prefix=self._evaluation_prefix,
            upsampling_factor=self._upsampling_factor,
        )
