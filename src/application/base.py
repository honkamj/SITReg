"""Base application implementations"""

from abc import abstractmethod
from time import time
from types import TracebackType
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, TypeVar

from composable_mapping import GridComposableMapping
from torch import Tensor
from torch import device as torch_device
from torch.cuda import (
    Event,
    current_stream,
    max_memory_allocated,
    reset_peak_memory_stats,
    synchronize,
)

from data.interface import InferenceMetadata, IStorage
from data.storage import (
    FloatStorage,
    OptionalStorageWrapper,
    SequenceStorageWrapper,
    StringStorage,
    TensorCompressedStorage,
    TorchStorage,
)
from util.device import get_device_name

from .interface import (
    ICaseInferenceDefinition,
    IInferenceDefinition,
    ITrainingDefinition,
)

T = TypeVar("T", bound="BaseCaseInferenceDefinition")


class BaseCaseInferenceDefinition(ICaseInferenceDefinition):
    """Base case inference implementation"""

    def __init__(self, device: torch_device) -> None:
        self._device = device
        if device.type == "cuda":
            self._events = (Event(enable_timing=True), Event(enable_timing=True))
            self._cuda_stream = current_stream(device=device)
        elif device.type == "cpu":
            self._start_time: float
        self._elapsed_time: float = float("nan")  # In seconds
        self._memory_usage: float = float("nan")  # In megabytes

    def __enter__(self: T) -> T:
        if self._device.type == "cpu":
            self._start_time = time()
        elif self._device.type == "cuda":
            reset_peak_memory_stats(self._device)
            self._events[0].record(self._cuda_stream)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._device.type == "cpu":
            self._elapsed_time = time() - self._start_time
        elif self._device.type == "cuda":
            self._events[1].record(self._cuda_stream)
            synchronize(self._device)
            self._elapsed_time = self._events[0].elapsed_time(self._events[1]) / 1000
            self._memory_usage = max_memory_allocated(self._device) / 2**20
        return False

    def get_outputs(self) -> Mapping[str, Any]:
        return {
            "inference_time": self._elapsed_time,
            "inference_memory_usage": self._memory_usage,
            "device_name": get_device_name(self._device),
        }


class BaseRegistrationCaseInferenceDefinition(BaseCaseInferenceDefinition):
    """Base case inference implementation for pair-wise registration"""

    def __init__(
        self,
        application_config: Mapping[str, Any],
        device: torch_device,
    ) -> None:
        super().__init__(device)
        self._do_reverse_inference: bool = application_config["inference"].get(
            "do_reverse_inference", False
        )
        self._images_1: list[Tensor | None] = []
        self._images_2: list[Tensor | None] = []
        self._resampled_images_1: list[Tensor | None] = []
        self._resampled_images_2: list[Tensor | None] = []
        self._forward_displacement_fields: list[Tensor | None] = []
        self._inverse_displacement_fields: list[Tensor | None] = []

        self._forward_mappings: list[Optional[GridComposableMapping]] = []
        self._inverse_mappings: list[Optional[GridComposableMapping]] = []

        self._save_as_composable_mapping: bool = bool(
            application_config["inference"].get("save_as_composable_mapping", False)
        )

    def infer(self, batch: Any) -> None:
        image_1, image_2 = batch
        (
            resampled_image_1,
            resampled_image_2,
            forward_displacement_field,
            inverse_displacement_field,
            forward_mapping,
            inverse_mapping,
        ) = self._infer(
            image_1=image_1,
            image_2=image_2,
        )
        self._images_1.append(image_1[0])
        self._images_2.append(image_2[0])
        self._resampled_images_1.append(resampled_image_1)
        self._resampled_images_2.append(resampled_image_2)
        self._forward_displacement_fields.append(forward_displacement_field)
        self._inverse_displacement_fields.append(inverse_displacement_field)
        if self._save_as_composable_mapping:
            self._forward_mappings.append(forward_mapping)
            self._inverse_mappings.append(inverse_mapping)
        if self._do_reverse_inference:
            (
                resampled_image_2,
                resampled_image_1,
                inverse_displacement_field,
                forward_displacement_field,
                inverse_mapping,
                forward_mapping,
            ) = self._infer(
                image_1=image_2,
                image_2=image_1,
            )
            self._images_1.append(image_1[0])
            self._images_2.append(image_2[0])
            self._resampled_images_1.append(resampled_image_1)
            self._resampled_images_2.append(resampled_image_2)
            self._forward_displacement_fields.append(forward_displacement_field)
            self._inverse_displacement_fields.append(inverse_displacement_field)
            if self._save_as_composable_mapping:
                self._forward_mappings.append(forward_mapping)
                self._inverse_mappings.append(inverse_mapping)

    @abstractmethod
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
        """Do registration between two images"""

    def get_outputs(self) -> Mapping[str, Any]:
        outputs: dict[str, Any] = {
            "image_1": self._images_1,
            "image_2": self._images_2,
            "image_1_resampled": self._resampled_images_1,
            "image_2_resampled": self._resampled_images_2,
            "forward_displacement_field": self._forward_displacement_fields,
            "inverse_displacement_field": self._inverse_displacement_fields,
        }
        if self._save_as_composable_mapping:
            outputs["forward_mapping"] = self._forward_mappings
            outputs["inverse_mapping"] = self._inverse_mappings
        return outputs | dict(super().get_outputs())


class BaseInferenceDefinition(IInferenceDefinition):
    """Base inference implementation"""

    def get_output_storages(self, inference_metadata: InferenceMetadata) -> Mapping[str, IStorage]:
        return {
            "inference_time": FloatStorage("inference_time"),
            "inference_memory_usage": FloatStorage("inference_memory_usage"),
            "device_name": StringStorage("device_name"),
        }


class BaseRegistrationInferenceDefinition(BaseInferenceDefinition):
    """Base inference implementation"""

    def __init__(
        self,
        application_config: Mapping[str, Any],
    ) -> None:
        self._application_config = application_config
        self._save_as_composable_mapping: bool = bool(
            application_config["inference"].get("save_as_composable_mapping", False)
        )

    def get_output_storages(self, inference_metadata: InferenceMetadata) -> Mapping[str, IStorage]:
        num_items = (
            2 if self._application_config["inference"].get("do_reverse_inference", False) else 1
        )
        output_storages = {
            "image_1": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    inference_metadata.default_storage_factories[0].create(
                        inference_metadata.names[0]
                    ),
                    name=inference_metadata.names[0],
                ),
                identifier="input_order",
                num_items=num_items,
            ),
            "image_2": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    inference_metadata.default_storage_factories[1].create(
                        inference_metadata.names[1]
                    ),
                    name=inference_metadata.names[1],
                ),
                identifier="input_order",
                num_items=num_items,
            ),
            "image_1_resampled": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    inference_metadata.default_storage_factories[0].create(
                        f"{inference_metadata.names[0]}_resampled"
                    ),
                    name=f"{inference_metadata.names[0]}_resampled",
                ),
                identifier="input_order",
                num_items=num_items,
            ),
            "image_2_resampled": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    inference_metadata.default_storage_factories[1].create(
                        f"{inference_metadata.names[1]}_resampled"
                    ),
                    name=f"{inference_metadata.names[1]}_resampled",
                ),
                identifier="input_order",
                num_items=num_items,
            ),
            "forward_displacement_field": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    TensorCompressedStorage(f"{inference_metadata.names[0]}_deformation"),
                    name=f"{inference_metadata.names[0]}_deformation",
                ),
                identifier="input_order",
                num_items=num_items,
            ),
            "inverse_displacement_field": SequenceStorageWrapper(
                OptionalStorageWrapper(
                    TensorCompressedStorage(f"{inference_metadata.names[1]}_deformation"),
                    name=f"{inference_metadata.names[1]}_deformation",
                ),
                identifier="input_order",
                num_items=num_items,
            ),
        }
        if self._save_as_composable_mapping:
            output_storages["forward_mapping"] = SequenceStorageWrapper(
                OptionalStorageWrapper(
                    TorchStorage(f"{inference_metadata.names[0]}_mapping"),
                    name=f"{inference_metadata.names[0]}_mapping",
                ),
                identifier="input_order",
                num_items=num_items,
            )
            output_storages["inverse_mapping"] = SequenceStorageWrapper(
                OptionalStorageWrapper(
                    TorchStorage(f"{inference_metadata.names[1]}_mapping"),
                    name=f"{inference_metadata.names[1]}_mapping",
                ),
                identifier="input_order",
                num_items=num_items,
            )
            output_storages["mapping_coordinate_system"] = SequenceStorageWrapper(
                OptionalStorageWrapper(
                    TorchStorage("mapping_coordinate_system"),
                    name="mapping_coordinate_system",
                ),
                identifier="input_order",
                num_items=num_items,
            )
        return output_storages | dict(super().get_output_storages(inference_metadata))


class BaseTrainingDefinition(ITrainingDefinition):
    """Base training definition"""

    def __init__(self, application_config: Mapping[str, Any]) -> None:
        self._n_epochs = application_config["training"]["n_epochs"]
        self._current_epoch = 0
        self._n_steps_per_epoch = 0

    def start_of_epoch(self, epoch: int, n_steps: int) -> None:
        self._current_epoch = epoch
        self._n_steps_per_epoch = n_steps

    @property
    def n_epochs(self) -> int:
        return self._n_epochs

    def displayed_metrics(self) -> Optional[Iterable[str]]:
        return None

    def get_custom_mass_functions(
        self,
    ) -> Optional[Mapping[str, Callable[[float, float], float]]]:
        return None

    def before_save(self, saving_process_rank: int) -> None:
        pass
