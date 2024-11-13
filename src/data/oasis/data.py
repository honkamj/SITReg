"""Oasis dataset"""

from itertools import combinations, permutations
from logging import getLogger
from os.path import join
from typing import Sequence, cast

from nibabel import load as nib_load  # type: ignore
from torch import Tensor, from_numpy, get_default_dtype, ones

from data.base import BaseDataDownloader, BaseVolumetricRegistrationData
from data.interface import VolumetricDataArgs
from util.download import download
from util.extract import untar

logger = getLogger(__name__)


class OasisDataDownloader(BaseDataDownloader):
    """Oasis data downloader"""

    DATASET_URL = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar"

    def __init__(self) -> None:
        super().__init__(dataset_name="oasis")

    def _get_license_agreement_question(self) -> str:
        return (
            "Neurite OASIS dataset from Learn2Reg challenge is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://www.oasis-brains.org/#access and "
            "https://learn2reg-test.grand-challenge.org/datasets/. "
            "(y/n) "
        )

    def _download_and_process(self, data_folder: str) -> None:
        dataset_tar_path = join(data_folder, "neurite-oasis.v1.0.tar")
        logger.info("Downloading OASIS dataset...")
        download(
            source_url=self.DATASET_URL,
            target_path=dataset_tar_path,
            description="Downloading OASIS dataset",
        )
        logger.info("Extracting OASIS dataset")
        untar(file_path=dataset_tar_path, extract_to_same_dir=True, remove_after=True)


class OasisData(BaseVolumetricRegistrationData):
    """Class for accessing OASIS cases"""

    def __init__(
        self,
        data_root: str,
        both_directions: bool,
        file_type: str,
        segmentation_file_type: str,
        included_segmentation_class_indices: Sequence[int] | None = None,
        training_segmentation_class_index_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            data_downloader=OasisDataDownloader(),
            included_segmentation_class_indices=included_segmentation_class_indices,
            training_segmentation_class_index_groups=training_segmentation_class_index_groups,
        )
        self._both_directions = both_directions
        self._image_file_type = file_type
        self._segmentation_file_type = segmentation_file_type

    def get_case_affine(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        return from_numpy(self._get_spatial_image_for_case(case_name, self._image_file_type).affine)

    def _get_raw_shape_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Sequence[int]:
        return self._get_spatial_image_for_case(case_name, self._image_file_type).shape

    def _get_raw_data_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        data = self._get_spatial_image_for_case(case_name, self._image_file_type).get_fdata()
        return from_numpy(data).to(get_default_dtype())[None]

    def _get_raw_segmentation_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        data = self._get_spatial_image_for_case(case_name, self._segmentation_file_type).dataobj[
            ...
        ]
        return from_numpy(data).to(get_default_dtype())

    def _get_raw_mask_for_case(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Tensor:
        shape = self._get_raw_shape_for_case(case_name, args, registration_index)
        return ones(shape)[None]

    def _get_path_to_case(self, case_name: str, file_type: str) -> str:
        """Get path to case file"""
        return join(self._data_location, case_name, f"{file_type}.nii.gz")

    def _get_pairs(self, cases: Sequence[str]) -> list[tuple[str, str]]:
        pair_iterator = permutations if self._both_directions else combinations
        return cast(list[tuple[str, str]], list(pair_iterator(cases, r=2)))

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return self._get_pairs(self._get_cases()[255:275])

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        return self._get_pairs(self._get_cases()[275:])

    def get_train_cases(self) -> Sequence[str]:
        """Get training cases"""
        return self._get_cases()[:255]

    def _get_cases(self) -> list[str]:
        with open(
            join(self._data_location, "subjects.txt"), mode="r", encoding="utf-8"
        ) as cases_file:
            return cases_file.read().splitlines()

    def _get_spatial_image_for_case(self, case_name: str, file_type: str):
        return nib_load(self._get_path_to_case(case_name, file_type))


class OasisDataLearn2Reg(OasisData):
    """OASIS data with Learn2Reg data splits"""

    VALIDATE_PAIRS = [
        ("OASIS_OAS1_0438_MR1", "OASIS_OAS1_0439_MR1"),
        ("OASIS_OAS1_0439_MR1", "OASIS_OAS1_0440_MR1"),
        ("OASIS_OAS1_0440_MR1", "OASIS_OAS1_0441_MR1"),
        ("OASIS_OAS1_0441_MR1", "OASIS_OAS1_0442_MR1"),
        ("OASIS_OAS1_0442_MR1", "OASIS_OAS1_0443_MR1"),
        ("OASIS_OAS1_0443_MR1", "OASIS_OAS1_0444_MR1"),
        ("OASIS_OAS1_0444_MR1", "OASIS_OAS1_0445_MR1"),
        ("OASIS_OAS1_0445_MR1", "OASIS_OAS1_0446_MR1"),
        ("OASIS_OAS1_0446_MR1", "OASIS_OAS1_0447_MR1"),
        ("OASIS_OAS1_0447_MR1", "OASIS_OAS1_0448_MR1"),
        ("OASIS_OAS1_0448_MR1", "OASIS_OAS1_0449_MR1"),
        ("OASIS_OAS1_0449_MR1", "OASIS_OAS1_0450_MR1"),
        ("OASIS_OAS1_0450_MR1", "OASIS_OAS1_0451_MR1"),
        ("OASIS_OAS1_0451_MR1", "OASIS_OAS1_0452_MR1"),
        ("OASIS_OAS1_0452_MR1", "OASIS_OAS1_0453_MR1"),
        ("OASIS_OAS1_0453_MR1", "OASIS_OAS1_0454_MR1"),
        ("OASIS_OAS1_0454_MR1", "OASIS_OAS1_0455_MR1"),
        ("OASIS_OAS1_0455_MR1", "OASIS_OAS1_0456_MR1"),
        ("OASIS_OAS1_0456_MR1", "OASIS_OAS1_0457_MR1"),
    ]

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return self.VALIDATE_PAIRS

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        return []

    def get_train_cases(self) -> Sequence[str]:
        all_validate_cases = set()
        for validate_pair in self.VALIDATE_PAIRS:
            all_validate_cases.add(validate_pair[0])
            all_validate_cases.add(validate_pair[1])
        return [case for case in self._get_cases() if case not in all_validate_cases]
