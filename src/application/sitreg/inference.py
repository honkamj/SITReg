"""SITReg registration inference implementation"""

from typing import Any, Mapping, Optional

from composable_mapping import (
    DataFormat,
    EnumeratedSamplingParameterCache,
    GridComposableMapping,
    LinearInterpolator,
    samplable_volume,
)
from torch import Tensor, cat
from torch import device as torch_device
from torch.nn import Module

from application.base import (
    BaseRegistrationCaseInferenceDefinition,
    BaseRegistrationInferenceDefinition,
)
from data.interface import InferenceMetadata, IStorage
from data.storage import SequenceStorageWrapper, TorchStorage
from model.sitreg import MappingPair, SITReg


class SITRegCaseInference(BaseRegistrationCaseInferenceDefinition):
    """Case inference for SITReg"""

    def __init__(
        self,
        model: SITReg,
        application_config: Mapping[str, Any],
        device: torch_device,
    ) -> None:
        super().__init__(application_config=application_config, device=device)
        self._model = model
        self._save_intermediate_mappings_for_levels: list[int] = (
            []
            if application_config["inference"].get("save_intermediate_mappings_for_levels") is None
            else application_config["inference"]["save_intermediate_mappings_for_levels"]
        )
        self._resample_when_composing: bool = bool(
            application_config["inference"].get("resample_when_composing", True)
        )
        self._intermediate_mappings: list[list[Optional[tuple[Tensor, Tensor]]]] = []
        self._sampling_parameter_cache = EnumeratedSamplingParameterCache()

    def _compute_mapping_pair(
        self,
        image_1: Tensor,
        image_2: Tensor,
    ) -> tuple[MappingPair, list[MappingPair]]:
        mapping_pair: MappingPair
        intermediate_mappings: list[MappingPair]
        mapping_pair, *intermediate_mappings = self._model(
            image_1,
            image_2,
            mappings_for_levels=[(0, True)]
            + [(level_index, True) for level_index in self._save_intermediate_mappings_for_levels],
            resample_when_composing=self._resample_when_composing,
        )
        return mapping_pair, intermediate_mappings

    def infer(self, batch: Any) -> None:
        with self._sampling_parameter_cache:
            return super().infer(batch)

    def _infer(
        self,
        image_1: Tensor,
        image_2: Tensor,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        GridComposableMapping | None,
        GridComposableMapping | None,
    ]:
        assert image_1.size(0) == 1 and image_2.size(0) == 1
        image_1 = image_1.to(self._device)
        image_2 = image_2.to(self._device)
        mapping_pair, intermediate_mappings = self._compute_mapping_pair(image_1, image_2)
        image_1_mapping = samplable_volume(
            image_1,
            coordinate_system=self._model.image_coordinate_system,
            sampler=LinearInterpolator(),
        )
        image_2_mapping = samplable_volume(
            image_2,
            coordinate_system=self._model.image_coordinate_system,
            sampler=LinearInterpolator(),
        )
        resampled_image_1 = (
            (image_1_mapping @ mapping_pair.forward_mapping).sample().generate_values()[0]
        )
        resampled_image_2 = (
            (image_2_mapping @ mapping_pair.inverse_mapping).sample().generate_values()[0]
        )
        forward_displacement_field = mapping_pair.forward_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()[0]
        inverse_displacement_field = mapping_pair.inverse_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()[0]
        if self._save_intermediate_mappings_for_levels:
            self._intermediate_mappings.append(
                [
                    (
                        intermediate_mappings[index]
                        .forward_mapping.sample(DataFormat.voxel_displacements())
                        .generate_values()[0],
                        intermediate_mappings[index]
                        .inverse_mapping.sample(DataFormat.voxel_displacements())
                        .generate_values()[0],
                    )
                    for index in range(len(intermediate_mappings))
                ]
            )
        return (
            resampled_image_1,
            resampled_image_2,
            forward_displacement_field,
            inverse_displacement_field,
            mapping_pair.forward_mapping,
            mapping_pair.inverse_mapping,
        )

    def get_outputs(self) -> Mapping[str, Any]:
        outputs: dict[str, Any] = {}
        if self._save_intermediate_mappings_for_levels:
            outputs["intermediate_mappings"] = self._intermediate_mappings
        return outputs | dict(super().get_outputs())


class SITRegInference(BaseRegistrationInferenceDefinition):
    """Inference for SITReg"""

    def __init__(
        self,
        model: SITReg,
        application_config: Mapping[str, Any],
        device: torch_device,
    ) -> None:
        super().__init__(application_config)
        self._device = device
        self._model = model
        self._application_config = application_config
        self._save_intermediate_mappings: bool = bool(
            application_config["inference"].get("save_intermediate_mappings_for_levels", False)
        )

    def get_modules(self) -> Mapping[str, Module]:
        return {"registration_network": self._model}

    def get_case_inference(self, inference_metadata: InferenceMetadata) -> SITRegCaseInference:
        return SITRegCaseInference(
            model=self._model,
            application_config=self._application_config,
            device=self._device,
        )

    def get_output_storages(self, inference_metadata: InferenceMetadata) -> Mapping[str, IStorage]:
        num_items = (
            2 if self._application_config["inference"].get("do_reverse_inference", False) else 1
        )
        output_storages = {}
        if self._save_intermediate_mappings:
            output_storages["intermediate_mappings"] = SequenceStorageWrapper(
                TorchStorage("intermediate_deformations"),
                identifier="input_order",
                num_items=num_items,
            )
        return output_storages | dict(super().get_output_storages(inference_metadata))


class SITRegCaseRegularityFineTuningInference(SITRegCaseInference):
    """Case regularity fine tuned inference for SITReg"""

    def __init__(
        self,
        model: SITReg,
        ndv_model: Module,
        application_config: Mapping[str, Any],
        device: torch_device,
    ) -> None:
        super().__init__(model=model, application_config=application_config, device=device)
        self._ndv_model = ndv_model

    def _compute_mapping_pair(
        self,
        image_1: Tensor,
        image_2: Tensor,
    ) -> tuple[MappingPair, list[MappingPair]]:
        assert image_1.size(0) == 1 and image_2.size(0) == 1
        image_1 = image_1.to(self._device)
        image_2 = image_2.to(self._device)
        mapping_pair: MappingPair
        intermediate_mappings: list[MappingPair]

        mapping_pair, *intermediate_mappings = self._model(
            image_1,
            image_2,
            mappings_for_levels=[(0, True)]
            + [(level_index, True) for level_index in self._save_intermediate_mappings_for_levels],
            resample_when_composing=self._resample_when_composing,
        )
        forward_ddf = mapping_pair.forward_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()
        inverse_ddf = mapping_pair.inverse_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()
        ddfs = cat([forward_ddf, inverse_ddf], dim=0)

        modification = self._ndv_model(ddfs)
        modified_ddfs: Tensor = ddfs + modification
        modified_forward_ddf, modified_inverse_ddf = modified_ddfs.chunk(2, dim=0)

        updated_forward_mapping = samplable_volume(
            modified_forward_ddf,
            coordinate_system=self._model.image_coordinate_system,
            data_format=DataFormat.voxel_displacements(),
            sampler=LinearInterpolator(),
        )
        updated_inverse_mapping = samplable_volume(
            modified_inverse_ddf,
            coordinate_system=self._model.image_coordinate_system,
            data_format=DataFormat.voxel_displacements(),
            sampler=LinearInterpolator(),
        )

        mapping_pair = MappingPair(
            forward_mapping=updated_forward_mapping,
            inverse_mapping=updated_inverse_mapping,
        )

        return mapping_pair, intermediate_mappings


class SITRegRegularityFineTuningInference(SITRegInference):
    """Regularity fine tuned inference for SITReg"""

    def __init__(
        self,
        model: SITReg,
        ndv_model: Module,
        application_config: Mapping[str, Any],
        device: torch_device,
    ) -> None:
        super().__init__(model=model, application_config=application_config, device=device)
        self._ndv_model = ndv_model

    def get_modules(self) -> Mapping[str, Module]:
        return {"registration_network": self._model, "ndv_network": self._ndv_model}

    def get_case_inference(
        self, inference_metadata: InferenceMetadata
    ) -> SITRegCaseRegularityFineTuningInference:
        return SITRegCaseRegularityFineTuningInference(
            model=self._model,
            ndv_model=self._ndv_model,
            application_config=self._application_config,
            device=self._device,
        )
