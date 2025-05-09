"""Oasis dataset"""

from itertools import combinations, permutations
from json import load as json_load
from logging import getLogger
from os import listdir, makedirs
from os.path import basename, join
from typing import Sequence, cast

import gdown  # type: ignore
from nibabel import load as nib_load  # type: ignore
from scipy.ndimage import binary_erosion  # type: ignore
from torch import Tensor, from_numpy, get_default_dtype, ones

from data.base import BaseDataDownloader, BaseVolumetricRegistrationData
from data.interface import VolumetricDataArgs
from util.extract import unzip

logger = getLogger(__name__)


class LumirDataDownloader(BaseDataDownloader):
    """Lumir data downloader"""

    def __init__(self) -> None:
        super().__init__(dataset_name="LUMIR")

    def _get_license_agreement_question(self) -> str:
        return (
            "By downloading the data you agree to the terms of use and the licence at "
            "https://learn2reg.grand-challenge.org/ Do you want to continue? (y/n)"
        )

    def _download_and_process(self, data_folder: str) -> None:
        logger.info("Downloading LUMIR dataset...")
        gdown.download(
            "https://drive.google.com/uc?export=download&id=1PTHAX9hZX7HBXXUGVvI1ar1LUf4aVbq9",
            join(data_folder, "LUMIR.zip"),
            quiet=False,
        )
        makedirs(join(data_folder, "labelsTr"))
        gdown.download(
            "https://drive.google.com/uc?export=download&id=14IQ_hiyMoheQqB_LrveDayzFaOe0YrEP",
            join(data_folder, "labelsTr", "SanityCheckLabelsTr.zip"),
            quiet=False,
        )
        gdown.download(
            "https://drive.google.com/uc?export=download&id=1b0hyH7ggjCysJG-VGvo38XVE8bFVRMxb",
            join(data_folder, "LUMIR_dataset.json"),
            quiet=False,
        )
        logger.info("Extracting LUMIR dataset...")
        unzip(join(data_folder, "LUMIR.zip"), extract_to_same_dir=True, remove_after=True)
        unzip(
            join(data_folder, "labelsTr", "SanityCheckLabelsTr.zip"),
            extract_to_same_dir=True,
            remove_after=True,
        )


class LumirData(BaseVolumetricRegistrationData):
    """Class for accessing Lumir cases"""

    def __init__(
        self,
        data_root: str,
        both_directions: bool,
        included_segmentation_class_indices: Sequence[int] | None = None,
        training_segmentation_class_index_groups: Sequence[Sequence[int]] | None = None,
        use_body_mask_as_mask_with_erosion: int | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            data_downloader=LumirDataDownloader(),
            included_segmentation_class_indices=included_segmentation_class_indices,
            training_segmentation_class_index_groups=training_segmentation_class_index_groups,
        )
        self._both_directions = both_directions
        with open(
            join(self._data_location, "LUMIR_dataset.json"), encoding="utf-8"
        ) as data_json_file:
            data_json = json_load(data_json_file)
        self._training_cases = [
            self._extract_case_name_from_path(item["image"]) for item in data_json["training"]
        ]
        self._validation_pairs = [
            (
                self._extract_case_name_from_path(item["moving"]),
                self._extract_case_name_from_path(item["fixed"]),
            )
            for item in data_json["validation"]
        ]
        self._sanity_check_validation_cases = [
            self._extract_case_name_from_path(file_name)
            for file_name in listdir(join(self._data_location, "labelsTr"))
        ]
        self._validation_cases = list(
            set([case for pair in self._validation_pairs for case in pair])
        )
        self._use_body_mask_as_mask_with_erosion = use_body_mask_as_mask_with_erosion

    @staticmethod
    def _extract_case_name_from_path(path: str) -> str:
        return basename(path).split(".")[0]

    def get_case_affine(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        return from_numpy(self._get_spatial_image_for_case(case_name).affine)

    def _get_raw_shape_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Sequence[int]:
        return self._get_spatial_image_for_case(case_name).shape

    def _get_raw_data_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        data = self._get_spatial_image_for_case(case_name).get_fdata()
        return from_numpy(data).to(get_default_dtype())[None]

    def _get_raw_segmentation_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        data = nib_load(self._get_path_to_seg(case_name)).get_fdata()  # type: ignore
        return from_numpy(data).to(get_default_dtype())

    def _get_raw_mask_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        if self._use_body_mask_as_mask_with_erosion is not None:
            data = self._get_spatial_image_for_case(case_name).get_fdata()
            body_mask = (data > 0).astype("float32")
            if self._use_body_mask_as_mask_with_erosion > 0:
                body_mask = binary_erosion(
                    body_mask,
                    iterations=self._use_body_mask_as_mask_with_erosion,
                )
            return from_numpy(body_mask).to(get_default_dtype())[None]
        shape = self._get_raw_shape_for_case(case_name, args, registration_index)
        return ones(shape)[None]

    def _get_path_to_image(self, case_name: str) -> str:
        if case_name in self._validation_cases:
            case_folder = "imagesVal"
        elif case_name in self._training_cases:
            case_folder = "imagesTr"
        else:
            raise ValueError(f"Case {case_name} not found in training or validation cases")
        return join(self._data_location, case_folder, f"{case_name}.nii.gz")

    def _get_path_to_seg(self, case_name: str) -> str:
        if case_name in self._sanity_check_validation_cases:
            return join(self._data_location, "labelsTr", f"{case_name}.nii.gz")
        raise ValueError(f"No segmentation available for case {case_name}")

    def get_inference_pairs(self, division: str) -> Sequence[tuple[str, str]]:
        if division == "sanity_check":
            return self._get_pairs(self._sanity_check_validation_cases)
        return super().get_inference_pairs(division)

    def _get_pairs(self, cases: Sequence[str]) -> list[tuple[str, str]]:
        pair_iterator = permutations if self._both_directions else combinations
        return cast(list[tuple[str, str]], list(pair_iterator(cases, r=2)))

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return self._validation_pairs

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        raise NotImplementedError("Test pairs are not available")

    def get_train_cases(self) -> Sequence[str]:
        return self._training_cases

    def _get_spatial_image_for_case(self, case_name: str):
        return nib_load(self._get_path_to_image(case_name))
