"""Interface for data"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence

from attr import define, field
from torch import Tensor
from torch import device as torch_device
from torch.utils.data import DataLoader, Dataset

from util.checked_type_casting import (
    to_optional_list_of_two_tuples,
    to_optional_two_tuple,
)
from util.import_util import import_object
from util.metrics import ISummarizer


class IVariantDataset(Dataset):
    """Base dataset which supports generating variants with multiprocessing"""

    @abstractmethod
    def generate_new_variant(self) -> None:
        """Generate new variant of the dataset"""

    @abstractmethod
    def __len__(self) -> int:
        """Number of items in the dataset"""


class IDataDownloader(ABC):
    """Data downloader interface"""

    @abstractmethod
    def download(self, data_root: str) -> str:
        """Download the dataset to data_root

        Returns: Path to the data folder
        """


class VolumetricDataArgs(NamedTuple):
    """Defines possible modifications to volumetric data"""

    downsampling_factor: Sequence[int] | None
    crop: Sequence[tuple[int, int]] | None
    normalize: bool = False
    shift_and_normalize: tuple[float, float] | None = None
    clip: tuple[float | None, float | None] | None = None
    crop_or_pad_to: Sequence[int] | None = None
    mask_threshold: float | None = 1 - 1e-5

    @classmethod
    def from_config(cls, data_config) -> "VolumetricDataArgs":
        """Create from config"""
        return cls(
            downsampling_factor=data_config["downsampling_factor"],
            crop=to_optional_list_of_two_tuples(int, data_config.get("crop")),
            normalize=data_config.get("normalize", False),
            shift_and_normalize=to_optional_two_tuple(
                float, data_config.get("shift_and_normalize")
            ),
            crop_or_pad_to=data_config.get("crop_or_pad_to"),
            clip=data_config.get("clip"),
        )

    def modified_copy(self, **kwargs) -> "VolumetricDataArgs":
        """Create modified copy"""
        return VolumetricDataArgs(
            downsampling_factor=(
                kwargs["downsampling_factor"]
                if "downsampling_factor" in kwargs
                else self.downsampling_factor
            ),
            crop=kwargs["crop"] if "crop" in kwargs else self.crop,
            normalize=kwargs["normalize"] if "normalize" in kwargs else self.normalize,
            shift_and_normalize=(
                kwargs["shift_and_normalize"]
                if "shift_and_normalize" in kwargs
                else self.shift_and_normalize
            ),
            clip=kwargs["clip"] if "clip" in kwargs else self.clip,
            crop_or_pad_to=(
                kwargs["crop_or_pad_to"] if "crop_or_pad_to" in kwargs else self.crop_or_pad_to
            ),
        )


class IVolumetricRegistrationData(ABC):
    """Interface for accessing volumetric subject-to-subject registration data"""

    @abstractmethod
    def get_case_shape(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Sequence[int]:
        """Get shape for the given case"""

    @abstractmethod
    def get_case_affine(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        """Get affine transformation for the given case"""

    @abstractmethod
    def get_case_volume(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        """Get volume for the given case"""

    @abstractmethod
    def get_case_training_segmentation(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        """Get training segmentation (one-hot encoded) for the given case"""

    @abstractmethod
    def get_case_evaluation_segmentation(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        """Get segmentation for the given case"""

    @abstractmethod
    def get_case_mask(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        """Get mask for the given case"""

    @abstractmethod
    def get_inference_pairs(self, division: str) -> Sequence[tuple[str, str]]:
        """Get inference pairs"""

    @abstractmethod
    def get_train_cases(self) -> Sequence[str]:
        """Get training cases"""


class IVolumetricRegistrationInferenceDataset(Dataset):
    """Oasis inference dataset"""

    @property
    @abstractmethod
    def division(self) -> str:
        """Return data division"""

    @property
    @abstractmethod
    def data(self) -> IVolumetricRegistrationData:
        """Return underlying OasisData object"""

    @property
    @abstractmethod
    def data_args(self) -> VolumetricDataArgs:
        """Return data args"""

    @abstractmethod
    def get_pair(self, index: int, data_args: VolumetricDataArgs) -> tuple[Tensor, Tensor]:
        """Get data pair with given index and data args"""

    @abstractmethod
    def __len__(self) -> int:
        """Size of the dataset"""

    @abstractmethod
    def shapes(self, index: int) -> tuple[Sequence[int], Sequence[int]]:
        """Get shapes of the images"""

    @abstractmethod
    def affines(self, index: int) -> tuple[Tensor, Tensor]:
        """Get affines of the images"""

    @abstractmethod
    def names(self, index: int) -> tuple[str, str]:
        """Get names of the images"""


class TrainingDataLoaderArgs(NamedTuple):
    """Args for creating training data loaders"""

    data_root: str
    num_workers: int
    training_process_rank: int
    training_process_local_rank: int
    n_training_processes: int
    n_local_training_processes: int


class TrainingDataLoader(NamedTuple):
    """Training data loader together with any additional requirements

    Arguments:
        data_loader: DataLoader which iterates over the training data
        generate_new_variant: Function which generates new variant of the data set,
          called at the end of each epoch
    """

    data_loader: DataLoader
    generate_new_variant: Optional[Callable[[], None]]


class IStorage(ABC):
    """Class which can be used for saving data to disk"""

    @abstractmethod
    def save(self, item: Any, target_folder: str) -> None:
        """Save data to storage"""

    @abstractmethod
    def load(self, target_folder: str, device: torch_device) -> Any:
        """Load data from storage"""

    @abstractmethod
    def exists(self, target_folder: str) -> bool:
        """Whether data is saved to the storage"""

    @abstractmethod
    def clear(self, target_folder: str) -> None:
        """Remove file related to the storage"""


class IStorageFactory(ABC):
    """Factory for creating storages based on target path"""

    @abstractmethod
    def create(self, name: str) -> IStorage:
        """Build the storage"""


@define
class InferenceMetadata:
    """Inference related metadata

    Arguments:
        inference_name: Name of the data being inferred
        info: Any additional info
        names: Name of the data items in batch
        default_storage_factories: Default storage factories
    """

    inference_name: str
    info: Mapping[str, Any]
    names: Sequence[str]
    default_storage_factories: Sequence[IStorageFactory] = field()

    @default_storage_factories.validator
    def _check_sequence_length(self, _attribute, sequence):
        if len(sequence) != len(self.names):
            raise ValueError("Number of default storage factories must match with number of names")


class Storable(NamedTuple):
    """Saveable item

    Saved using the saver, or if it is not given, using the default saver of the
    InferenceDataLoader
    """

    data: Any
    name: str
    metadata: InferenceMetadata


class IEvaluator(ABC):
    """Interface for evaluation functions"""

    @abstractmethod
    def __call__(self, inference_outputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Calculate metrics for inference outputs"""

    @property
    @abstractmethod
    def evaluation_inference_outputs(self) -> set[str]:
        """Get inference outputs useful for evaluation"""


class IInferenceFactory(ABC):
    """Interface for classes generating inference data loaders"""

    @abstractmethod
    def __len__(self) -> int:
        """Number of cases"""

    @abstractmethod
    def get_metadata(self, index: int) -> InferenceMetadata:
        """Get inference metadata for index"""

    @abstractmethod
    def get_data_loader(self, index: int, num_workers: int) -> DataLoader:
        """Generate data loader for doing inference of one case"""

    @abstractmethod
    def get_evaluator(self, index: int, device: torch_device) -> IEvaluator:
        """Generate evaluator for case with index"""

    @abstractmethod
    def get_evaluator_summarizers(
        self,
    ) -> Mapping[str | None, Iterable[ISummarizer]]:
        """Get summarizers for computing metrics

        Corresponds to summarizers argument of util.metrics.MetricsGatherer
        """

    @abstractmethod
    def generate_dummy_batch_and_metadata(self) -> tuple[Any, InferenceMetadata]:
        """Generate dummy batch with same shape as the actual data and metadata"""


class InferenceDataArgs(NamedTuple):
    """Args for creating inference data loaders"""

    data_root: str
    division: str


def create_training_data_loader(
    config: Mapping[str, Any], args: TrainingDataLoaderArgs
) -> TrainingDataLoader:
    """Create training data loader based on config"""
    data_loader_module = config["data"]["module"]
    data_loader_config = config["data"]["config"]
    create_function = import_object(
        f"data.{data_loader_module}.interface.create_training_data_loader"
    )
    return create_function(data_loader_config, args)


def create_inference_data_factory(
    config: Mapping[str, Any], args: InferenceDataArgs
) -> IInferenceFactory:
    """Create inference data loader factory based on config"""
    data_module = config["data"]["module"]
    data_config = config["data"]["config"]
    create_function = import_object(f"data.{data_module}.interface.create_inference_data_factory")
    return create_function(data_config, args)
