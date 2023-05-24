"""SITReg registration inference implementation"""


from typing import Any, Mapping, Optional

from torch import Tensor
from torch import device as torch_device
from torch.nn import Module

from algorithm.composable_mapping.factory import ComposableFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs, as_displacement_field
from algorithm.interpolator import LinearInterpolator
from application.base import (
    BaseRegistrationCaseInferenceDefinition,
    BaseRegistrationInferenceDefinition,
)
from data.interface import InferenceMetadata, IStorage
from data.storage import SequenceStorageWrapper, TorchStorage
from model.sitreg import SITReg, MappingPair


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
        self._intermediate_mappings: list[list[Optional[tuple[Tensor, Tensor]]]] = []

    def _infer(
        self,
        image_1: Tensor,
        image_2: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        assert image_1.size(0) == 1 and image_2.size(0) == 1
        image_1 = image_1.to(self._device)
        image_2 = image_2.to(self._device)
        dtype = image_1.dtype
        mapping_pair: MappingPair
        intermediate_mappings: list[MappingPair]
        mapping_pair, *intermediate_mappings = self._model(
            image_1,
            image_2,
            mappings_for_levels=[(0, True)]
            + [(level_index, True) for level_index in self._save_intermediate_mappings_for_levels],
        )
        coordinate_system = self._model.image_coordinate_system
        image_1_mapping = ComposableFactory.create_volume(
            data=image_1,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=True
            ),
        )
        image_2_mapping = ComposableFactory.create_volume(
            data=image_2,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=True
            ),
        )
        resampled_image_1_masked = image_1_mapping.compose(mapping_pair.forward_mapping)(
            coordinate_system.grid
        )
        resampled_image_2_masked = image_2_mapping.compose(mapping_pair.inverse_mapping)(
            coordinate_system.grid
        )
        resampled_image_1 = resampled_image_1_masked.generate_values()[0]
        resampled_image_2 = resampled_image_2_masked.generate_values()[0]
        forward_displacement_field = as_displacement_field(
            mapping=mapping_pair.forward_mapping,
            coordinate_system=coordinate_system,
            device=self._device,
            dtype=dtype,
        )[0][0]
        inverse_displacement_field = as_displacement_field(
            mapping=mapping_pair.inverse_mapping,
            coordinate_system=coordinate_system,
            device=self._device,
            dtype=dtype,
        )[0][0]
        if self._save_intermediate_mappings_for_levels:
            self._intermediate_mappings.append(
                [
                    (
                        as_displacement_field(
                            mapping=intermediate_mappings[index].forward_mapping,
                            coordinate_system=coordinate_system,
                            device=self._device,
                            dtype=dtype,
                        )[0][0],
                        as_displacement_field(
                            mapping=intermediate_mappings[index].inverse_mapping,
                            coordinate_system=coordinate_system,
                            device=self._device,
                            dtype=dtype,
                        )[0][0],
                    )
                    for index in range(len(intermediate_mappings))
                ]
            )
        return (
            resampled_image_1,
            resampled_image_2,
            forward_displacement_field,
            inverse_displacement_field,
        )

    def get_outputs(self) -> Mapping[str, Any]:
        outputs: dict[str, Any] = {}
        if self._save_intermediate_mappings_for_levels:
            outputs["intermediate_mappings"] = self._intermediate_mappings
        return outputs | super().get_outputs()


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
        return output_storages | super().get_output_storages(inference_metadata)
