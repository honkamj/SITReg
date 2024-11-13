"""Inference data related implementations"""

from typing import Any, Mapping, cast

from torch import Tensor
from torch import device as torch_device

from data.dataset import VolumetricRegistrationInferenceDataset
from data.interface import InferenceMetadata
from data.lung250m_4b.data import Lung250M4BData
from data.lung250m_4b.evaluation import Lung250M4BEvaluator
from data.storage import NiftiStorageFactory

from ..base import BaseVolumetricRegistrationInferenceFactory


class Lung250M4BInferenceFactory(BaseVolumetricRegistrationInferenceFactory):
    """Lung250M-4B inference factory"""

    def __init__(
        self,
        dataset: VolumetricRegistrationInferenceDataset,
        data_config: Mapping[str, Any],
    ) -> None:
        super().__init__(dataset)
        self._metrics_to_compute = data_config["metrics"][dataset.division]
        self._n_jacobian_samples = data_config.get("n_jacobian_samples_in_evaluation")
        self._evaluation_prefix = data_config.get("evaluation_prefix", "")
        self._jacobian_sampling_base_seed = data_config.get("jacobian_sampling_base_seed", None)
        self._upsampling_factor = data_config.get("downsampling_factor", None)

    def get_metadata(self, index: int) -> InferenceMetadata:
        image_1_name, image_2_name = self._dataset.names(index)
        image_1_shape, image_2_shape = self._dataset.shapes(index)
        image_1_affine, image_2_affine = self._dataset.affines(index)
        return InferenceMetadata(
            inference_name=f"{image_1_name}_inhale-{image_2_name}_exhale",
            names=[f"{image_1_name}_inhale", f"{image_2_name}_exhale"],
            info={
                "image_1_shape": image_1_shape,
                "image_2_shape": image_2_shape,
                "image_1_affine": image_1_affine,
                "image_2_affine": image_2_affine,
            },
            default_storage_factories=[
                self._get_storage_factory(image_1_affine),
                self._get_storage_factory(image_2_affine),
            ],
        )

    def _get_storage_factory(self, affine: Tensor) -> NiftiStorageFactory:
        return NiftiStorageFactory(affine)

    def get_evaluator(self, index: int, device: torch_device) -> Lung250M4BEvaluator:
        image_1_name, image_2_name = self._dataset.names(index)
        data: Lung250M4BData = cast(Lung250M4BData, self._dataset.data)
        return Lung250M4BEvaluator(
            source_landmarks=data.get_case_landmarks(
                image_1_name,
                args=self._dataset.data_args,
                registration_index=0,
            ),
            target_landmarks=data.get_case_landmarks(
                image_2_name,
                args=self._dataset.data_args,
                registration_index=1,
            ),
            metrics_to_compute=self._metrics_to_compute,
            n_jacobian_samples=self._n_jacobian_samples,
            jacobian_sampling_seed=self._jacobian_sampling_base_seed + index
            if self._jacobian_sampling_base_seed is not None
            else None,
            evaluation_prefix=self._evaluation_prefix,
            upsampling_factor=self._upsampling_factor,
        )
