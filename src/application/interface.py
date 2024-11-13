"""Interface applications"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence

from torch import device as torch_device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data.interface import InferenceMetadata, IStorage
from util.import_util import import_object


class ITrainingDefinition(ABC):
    """Defines training"""

    @abstractmethod
    def update_weights(self, batch: Any) -> Mapping[str, float | tuple[float, float]]:
        """Update weights of the model based on data batch

        Returns loss or metric values to save and display.
        """

    @abstractmethod
    def start_of_epoch(self, epoch: int, n_steps: int) -> None:
        """Executed at the start of epoch"""

    @abstractmethod
    def before_save(self, saving_process_rank: int) -> None:
        """Executed before saving the model at the process of given rank

        Allows for e.g. consolidating zero redundancy optimizer state
        to the main process."""

    @abstractmethod
    def get_optimizers(self) -> Mapping[str, Optimizer | LRScheduler]:
        """Get optimizers"""

    @abstractmethod
    def get_modules(self) -> Mapping[str, Module]:
        """Get modules"""

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Number of epochs"""

    @abstractmethod
    def get_custom_mass_functions(self) -> Optional[Mapping[str, Callable[[float, float], float]]]:
        """Get custom mass functions for loss function accumulation

        Corresponds to custom_mass_functions argument of util.metrics.LossAverager
        """

    @abstractmethod
    def displayed_metrics(self) -> Optional[Iterable[str]]:
        """Which metric returned by update weights to display during training"""


class ICaseInferenceDefinition(AbstractContextManager):
    """Defines inference for a case"""

    @abstractmethod
    def __enter__(self) -> "ICaseInferenceDefinition":
        """Enter context manager"""

    @abstractmethod
    def infer(self, batch: Any) -> None:
        """Do inferece based on batch of data"""

    @abstractmethod
    def get_outputs(self) -> Mapping[str, Any]:
        """Get outputs to save"""


class IInferenceDefinition(ABC):
    """Defines inference"""

    @abstractmethod
    def get_modules(self) -> Mapping[str, Module]:
        """Get modules"""

    @abstractmethod
    def get_case_inference(self, inference_metadata: InferenceMetadata) -> ICaseInferenceDefinition:
        """Get inferecer for one case"""

    @abstractmethod
    def get_output_storages(self, inference_metadata: InferenceMetadata) -> Mapping[str, IStorage]:
        """Get storages for inference outputs"""


class TrainingDefinitionArgs(NamedTuple):
    """Args for creating training definitions"""

    devices: Sequence[torch_device]
    training_process_rank: int
    training_process_local_rank: int
    n_training_processes: int
    n_local_training_processes: int


def create_training_definition(
    config: Mapping[str, Any], args: TrainingDefinitionArgs
) -> ITrainingDefinition:
    """Create training definition based on config"""
    training_factory = config["application"]["training_factory"]
    application_config = config["application"]["config"]
    training_factory = import_object(config["application"]["training_factory"])
    return training_factory(application_config, args)


def create_inference_definition(
    config: Mapping[str, Any], device: torch_device
) -> IInferenceDefinition:
    """Create inference definition based on config"""
    application_config = config["application"]["config"]
    inference_factory = import_object(config["application"]["inference_factory"])
    return inference_factory(application_config, device)
