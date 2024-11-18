"""Dataset implementatoins"""

from math import perm
from typing import Any, NamedTuple, Sequence

from composable_mapping import Affine, ComposableMapping, Identity
from torch import Generator, Tensor, get_default_dtype, rand, randint, randperm, tensor

from algorithm.affine_sampling import (
    AffineTransformationSamplingArguments,
    sample_random_affine_transformation,
)
from algorithm.nth_permutation import nth_permutation_indices
from data.base import BaseVariantDataset
from data.interface import (
    IVariantDataset,
    IVolumetricRegistrationData,
    IVolumetricRegistrationInferenceDataset,
    VolumetricDataArgs,
)


class Augmentation(NamedTuple):
    """Augmentation"""

    type: str | Sequence[str]
    parameters: Tensor

    def as_mapping(self) -> ComposableMapping:
        """Return augmentation as composable mapping"""
        if isinstance(self.type, str):
            raise ValueError("Collate the augmentations first before converting to mapping.")
        if all(t == "identity" for t in self.type):
            return Identity(dtype=self.parameters.dtype)
        if all(t == "affine" for t in self.type):
            return Affine.from_matrix(self.parameters)
        raise ValueError(f"Unknown augmentation type: {self.type}")


class VolumetricRegistrationTrainingDataset(BaseVariantDataset):
    """Volumetric registration training dataset"""

    def __init__(
        self,
        data: IVolumetricRegistrationData,
        data_args: VolumetricDataArgs,
        seed: int,
        pairs_per_epoch: int | None = None,
        n_training_cases: int | None = None,
        affine_augmentation_arguments: AffineTransformationSamplingArguments | None = None,
        affine_augmentation_prob: float | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._data = data
        self._training_cases = data.get_train_cases()
        if n_training_cases is not None:
            self._training_cases = self._training_cases[:n_training_cases]
        self._n_pairs = self._compute_num_pairs(len(self._training_cases))
        self._data_args = data_args
        self._pairs_per_epoch = self._n_pairs if pairs_per_epoch is None else pairs_per_epoch
        self._affine_augmentation_arguments = affine_augmentation_arguments
        self._affine_augmentation_prob = affine_augmentation_prob
        self._pair_permutations: dict[int, Tensor] = {}
        self._affine_augmentations: Sequence[Sequence[Tensor | None]]
        self._cases_order: Tensor

    @staticmethod
    def _compute_num_pairs(n_training_cases: int) -> int:
        return perm(n_training_cases, 2)

    def _get_training_pair_index(self, index: int, generation: int) -> int:
        global_index = generation * self._pairs_per_epoch + index
        permutation_cycle, pair_index = divmod(global_index, self._n_pairs)
        return int(self._pair_permutations[permutation_cycle][pair_index])

    def _get_training_pair(self, index: int, generation: int) -> tuple[str, str]:
        training_pair_indices = nth_permutation_indices(
            len(self._training_cases),
            index=self._get_training_pair_index(index, generation),
            n_elements_per_permutation=2,
        )
        assert len(training_pair_indices) == 2
        return (
            self._training_cases[training_pair_indices[0]],
            self._training_cases[training_pair_indices[1]],
        )

    def _generate_new_variant(self, random_generator: Generator, generation: int) -> None:
        min_cycle_index = self._pairs_per_epoch * generation // self._n_pairs
        max_cycle_index = (self._pairs_per_epoch * (generation + 1) - 1) // self._n_pairs
        for existing_cycle_index in list(self._pair_permutations.keys()):
            if existing_cycle_index < min_cycle_index:
                self._pair_permutations.pop(existing_cycle_index)
        for cycle_index in range(min_cycle_index, max_cycle_index + 1):
            if cycle_index not in self._pair_permutations:
                self._pair_permutations[cycle_index] = randperm(
                    n=self._n_pairs, generator=random_generator
                )
        self._affine_augmentations = _sample_affines(
            affine_augmentation_arguments=self._affine_augmentation_arguments,
            affine_augmentation_prob=self._affine_augmentation_prob,
            random_generator=random_generator,
            n_affines=self._pairs_per_epoch,
        )

    def _length(self, generation: int) -> int:
        return self._pairs_per_epoch

    def _get_item(self, index: int, generation: int) -> Any:
        (
            first_case_name,
            second_case_name,
        ) = self._get_training_pair(index, generation)
        first_case = self._data.get_case_volume(
            first_case_name, args=self._data_args, registration_index=0
        )
        first_case_mask = self._data.get_case_mask(
            first_case_name, args=self._data_args, registration_index=0
        )
        second_case = self._data.get_case_volume(
            second_case_name, args=self._data_args, registration_index=1
        )
        second_case_mask = self._data.get_case_mask(
            second_case_name, args=self._data_args, registration_index=1
        )
        first_augmentation_parameters = self._affine_augmentations[index][0]
        first_augmentation = (
            Augmentation("identity", tensor([]))
            if first_augmentation_parameters is None
            else Augmentation("affine", first_augmentation_parameters)
        )
        second_augmentation_parameters = self._affine_augmentations[index][1]
        second_augmentation = (
            Augmentation("identity", tensor([]))
            if second_augmentation_parameters is None
            else Augmentation("affine", second_augmentation_parameters)
        )
        return (
            (first_case, first_case_mask, first_augmentation),
            (second_case, second_case_mask, second_augmentation),
        )


class VolumetricRegistrationSegmentationTrainingDataset(VolumetricRegistrationTrainingDataset):
    """Training dataset with segmentation maps"""

    def _get_item(self, index: int, generation: int) -> Any:
        item_1, item_2 = super()._get_item(index, generation)
        (
            first_case_name,
            second_case_name,
        ) = self._get_training_pair(index, generation)
        first_segmentation = self._data.get_case_training_segmentation(
            first_case_name, args=self._data_args, registration_index=0
        )
        second_segmentation = self._data.get_case_training_segmentation(
            second_case_name, args=self._data_args, registration_index=1
        )
        return (
            item_1 + (first_segmentation,),
            item_2 + (second_segmentation,),
        )


class VolumetricRegistrationLandmarkTrainingDataset(VolumetricRegistrationTrainingDataset):
    """Training dataset with landmarks"""

    def _get_item(self, index: int, generation: int) -> Any:
        item_1, item_2 = super()._get_item(index, generation)
        (
            first_case_name,
            second_case_name,
        ) = self._get_training_pair(index, generation)
        item_1 = tuple(item_1)
        item_2 = tuple(item_2)
        landmarks_1 = self._data.get_case_landmarks(
            first_case_name, args=self._data_args, registration_index=0
        )
        landmarks_2 = self._data.get_case_landmarks(
            second_case_name, args=self._data_args, registration_index=1
        )
        if landmarks_1 is None or landmarks_2 is None:
            raise ValueError("Landmarks are not available")
        return item_1 + (landmarks_1,), item_2 + (landmarks_2,)


class IntraCaseVolumetricRegistrationTrainingDataset(VolumetricRegistrationTrainingDataset):
    """Volumetric registration training dataset where registration is performed
    between two images of same case"""

    @staticmethod
    def _compute_num_pairs(n_training_cases: int) -> int:
        return n_training_cases

    def _get_training_pair(self, index: int, generation: int) -> tuple[str, str]:
        training_pair_index = self._get_training_pair_index(index, generation)
        return (
            self._training_cases[training_pair_index],
            self._training_cases[training_pair_index],
        )


class IntraCaseVolumetricRegistrationLandmarkTrainingDataset(
    VolumetricRegistrationLandmarkTrainingDataset
):
    """Volumetric registration training dataset where registration is performed
    between two images of same case together with landmarks"""

    @staticmethod
    def _compute_num_pairs(n_training_cases: int) -> int:
        return n_training_cases

    def _get_training_pair(self, index: int, generation: int) -> tuple[str, str]:
        training_pair_index = self._get_training_pair_index(index, generation)
        return (
            self._training_cases[training_pair_index],
            self._training_cases[training_pair_index],
        )


class VolumetricRegistrationInferenceDataset(IVolumetricRegistrationInferenceDataset):
    """Inference dataset"""

    def __init__(
        self,
        data: IVolumetricRegistrationData,
        data_args: VolumetricDataArgs,
        division: str,
    ) -> None:
        self._data = data
        self._data_args = data_args
        self._pairs = data.get_inference_pairs(division)
        self._division = division

    @property
    def division(self) -> str:
        """Return data division"""
        return self._division

    @property
    def data(self) -> IVolumetricRegistrationData:
        """Return underlying OasisData object"""
        return self._data

    @property
    def data_args(self) -> VolumetricDataArgs:
        """Return data args"""
        return self._data_args

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_pair(index, self._data_args)

    def get_pair(self, index: int, data_args: VolumetricDataArgs) -> tuple[Tensor, Tensor]:
        """Get data pair with given index and data args"""
        first_case_name, second_case_name = self._pairs[index]
        first_case = self._data.get_case_volume(
            first_case_name, args=data_args, registration_index=0
        )
        second_case = self._data.get_case_volume(
            second_case_name, args=data_args, registration_index=1
        )
        return first_case, second_case

    def shapes(self, index: int) -> tuple[Sequence[int], Sequence[int]]:
        """Get shapes of the images"""
        first_case_name, second_case_name = self._pairs[index]
        return (
            self._data.get_case_shape(first_case_name, args=self._data_args, registration_index=0),
            self._data.get_case_shape(second_case_name, args=self._data_args, registration_index=1),
        )

    def affines(self, index: int) -> tuple[Tensor, Tensor]:
        """Get affines of the images"""
        first_case_name, second_case_name = self._pairs[index]
        return (
            self._data.get_case_affine(first_case_name, args=self._data_args, registration_index=0),
            self._data.get_case_affine(
                second_case_name, args=self._data_args, registration_index=1
            ),
        )

    def names(self, index: int) -> tuple[str, str]:
        """Get names of the images"""
        return self._pairs[index]


class VolumetricRegistrationTrainingDatasetWithReplacement(BaseVariantDataset):
    """Volumetric registration training dataset where permutations are sampled with replacement

    This dataset is used for training with replacement, where each permutation
    can occur mutiple times before any given other permuation occurs. Suitable
    for training with large number of cases.
    """

    def __init__(
        self,
        data: IVolumetricRegistrationData,
        data_args: VolumetricDataArgs,
        seed: int,
        n_steps_per_epoch: int,
        n_training_cases: int | None = None,
        affine_augmentation_arguments: AffineTransformationSamplingArguments | None = None,
        affine_augmentation_prob: float | None = None,
        n_items_per_step: int = 2,
    ) -> None:
        super().__init__(seed=seed)
        self._data = data
        self._training_cases = data.get_train_cases()
        if n_training_cases is not None:
            self._training_cases = self._training_cases[:n_training_cases]
        self.n_items_per_step = n_items_per_step
        self._n_permutations = self._compute_num_permutations()
        self._permutation_indices: Tensor
        self._data_args = data_args
        self._n_steps_per_epoch = n_steps_per_epoch
        self._affine_augmentation_arguments = affine_augmentation_arguments
        self._affine_augmentation_prob = affine_augmentation_prob
        self._pair_permutations: dict[int, Tensor] = {}
        self._affine_augmentations: Sequence[Sequence[Tensor | None]]

    def _compute_num_permutations(self) -> int:
        return perm(len(self._training_cases), self.n_items_per_step)

    def _get_training_item(self, index: int) -> tuple[str, ...]:
        training_item_indices = nth_permutation_indices(
            len(self._training_cases),
            index=int(self._permutation_indices[index].item()),
            n_elements_per_permutation=self.n_items_per_step,
        )
        return tuple(self._training_cases[index] for index in training_item_indices)

    def _generate_new_variant(self, random_generator: Generator, generation: int) -> None:
        self._permutation_indices = randint(
            low=0,
            high=self._n_permutations,
            size=(self._n_steps_per_epoch,),
            generator=random_generator,
        )
        self._affine_augmentations = _sample_affines(
            affine_augmentation_arguments=self._affine_augmentation_arguments,
            affine_augmentation_prob=self._affine_augmentation_prob,
            random_generator=random_generator,
            n_affines=self._n_steps_per_epoch,
            n_items_per_step=self.n_items_per_step,
        )

    def _length(self, generation: int) -> int:
        return self._n_steps_per_epoch

    def _get_item(self, index: int, generation: int) -> Any:
        names = self._get_training_item(index)
        cases = []
        for index, (name, affine_augmentation) in enumerate(
            zip(names, self._affine_augmentations[index])
        ):
            volume = self._data.get_case_volume(
                name, args=self._data_args, registration_index=index
            )
            mask = self._data.get_case_mask(name, args=self._data_args, registration_index=index)
            cases.append((volume, mask, affine_augmentation))
        return tuple(cases)


class CyclicRegistrationDistributedDatasetAdapter(IVariantDataset):
    """Volumetric registration training dataset where permutations are sampled with replacement

    This dataset is used for training with replacement, where each permutation
    can occur mutiple times before any given other permuation occurs. Suitable
    for training with large number of cases.
    """

    def __init__(
        self,
        dataset: IVariantDataset,
        n_items_per_step: int,
        process_rank: int,
        include_first_mask_for_all: bool = False,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._process_rank = process_rank
        self._n_items_per_step = n_items_per_step
        self._include_first_mask_for_all = include_first_mask_for_all

    def generate_new_variant(self) -> None:
        """Generate new variant"""
        self._dataset.generate_new_variant()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        items = self._dataset[index]
        first_item_index = self._process_rank % self._n_items_per_step
        second_item_index = (self._process_rank + 1) % self._n_items_per_step
        output: tuple = items[first_item_index], items[second_item_index]
        if self._include_first_mask_for_all:
            output += (items[0][1],)
        return output


def _sample_affines(
    affine_augmentation_arguments: AffineTransformationSamplingArguments | None,
    affine_augmentation_prob: float | None,
    random_generator: Generator,
    n_affines: int,
    n_items_per_step: int = 2,
) -> Sequence[Sequence[Tensor | None]]:
    """Sample affines for augmentation"""
    affine_augmentations = []
    for _ in range(n_affines):
        affines: list[Tensor | None] = []
        for _ in range(n_items_per_step):
            if (
                affine_augmentation_arguments is not None
                and affine_augmentation_prob is not None
                and float(rand(size=(1,), generator=random_generator)[0]) < affine_augmentation_prob
            ):
                affines.append(
                    sample_random_affine_transformation(
                        n_transformations=1,
                        arguments=affine_augmentation_arguments,
                        generator=random_generator,
                        dtype=get_default_dtype(),
                    )[0]
                )
            else:
                affines.append(None)
        affine_augmentations.append(tuple(affines))
    return affine_augmentations
