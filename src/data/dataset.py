"""Dataset implementatoins"""

from math import perm
from typing import Sequence

from torch import Generator, Tensor, randperm

from algorithm.nth_permutation import nth_permutation_indices
from data.base import BaseVariantDataset
from data.interface import (
    IVolumetricRegistrationData,
    IVolumetricRegistrationInferenceDataset,
    VolumetricDataArgs,
)


class VolumetricRegistrationTrainingDataset(BaseVariantDataset):
    """Volumetric registration training dataset"""

    def __init__(
        self,
        data: IVolumetricRegistrationData,
        data_args: VolumetricDataArgs,
        seed: int,
        pairs_per_epoch: int | None = None,
        n_training_cases: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._data = data
        self._training_cases = data.get_train_cases()
        if n_training_cases is not None:
            self._training_cases = self._training_cases[:n_training_cases]
        self._n_pairs = perm(len(self._training_cases), 2)
        self._data_args = data_args
        self._pairs_per_epoch = self._n_pairs if pairs_per_epoch is None else pairs_per_epoch
        self._pair_permutations: dict[int, Tensor] = {}
        self._cases_order: Tensor

    def _get_training_pair(self, index: int, generation: int) -> tuple[str, str]:
        global_index = generation * self._pairs_per_epoch + index
        permutation_cycle, pair_index = divmod(global_index, self._n_pairs)
        training_pair_indices = nth_permutation_indices(
            len(self._training_cases),
            index=int(self._pair_permutations[permutation_cycle][pair_index]),
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

    def _length(self, generation: int) -> int:
        return self._pairs_per_epoch

    def _get_item(
        self, index: int, generation: int
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        (
            first_case_name,
            second_case_name,
        ) = self._get_training_pair(index, generation)
        first_case = self._data.get_case_volume(first_case_name, args=self._data_args)
        first_case_mask = self._data.get_case_mask(first_case_name, args=self._data_args)
        second_case = self._data.get_case_volume(second_case_name, args=self._data_args)
        second_case_mask = self._data.get_case_mask(second_case_name, args=self._data_args)
        return (
            (first_case[None], first_case_mask[None]),
            (second_case[None], second_case_mask[None]),
        )


class VolumetricRegistrationInferenceDataset(IVolumetricRegistrationInferenceDataset):
    """Oasis inference dataset"""

    def __init__(
        self, data: IVolumetricRegistrationData, data_args: VolumetricDataArgs, division: str
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
        first_case = self._data.get_case_volume(first_case_name, args=data_args)
        second_case = self._data.get_case_volume(second_case_name, args=data_args)
        return first_case[None], second_case[None]

    def shapes(self, index: int) -> tuple[Sequence[int], Sequence[int]]:
        """Get shapes of the images"""
        first_case_name, second_case_name = self._pairs[index]
        return (
            self._data.get_case_shape(first_case_name, args=self._data_args),
            self._data.get_case_shape(second_case_name, args=self._data_args),
        )

    def affines(self, index: int) -> tuple[Tensor, Tensor]:
        """Get affines of the images"""
        first_case_name, second_case_name = self._pairs[index]
        return (
            self._data.get_case_affine(first_case_name, args=self._data_args),
            self._data.get_case_affine(second_case_name, args=self._data_args),
        )

    def names(self, index: int) -> tuple[str, str]:
        """Get names of the images"""
        return self._pairs[index]
