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

    def __init__(self, data_root: str, both_directions: bool) -> None:
        super().__init__(data_root, OasisDataDownloader())
        self._both_directions = both_directions

    def get_case_affine(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        return from_numpy(self._get_spatial_image_for_case(case_name, args.file_type).affine)

    def _get_raw_shape_for_case(self, case_name: str, args: VolumetricDataArgs) -> Sequence[int]:
        return self._get_spatial_image_for_case(case_name, args.file_type).shape

    def _get_raw_data_for_case(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        data = self._get_spatial_image_for_case(case_name, args.file_type).get_fdata()
        return from_numpy(data).to(get_default_dtype())

    def _get_raw_mask_for_case(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        shape = self._get_raw_shape_for_case(case_name, args)
        return ones(shape)

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
