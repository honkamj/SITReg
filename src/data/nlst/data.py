"""Lung250M-4B dataset"""

from json import load as json_load
from logging import getLogger
from math import copysign
from os.path import join
from typing import Sequence

from nibabel import load as nib_load  # type: ignore
from numpy import loadtxt
from torch import Tensor, from_numpy, get_default_dtype

from data.base import BaseDataDownloader, BaseVolumetricRegistrationData
from data.interface import VolumetricDataArgs
from util.download import download
from util.extract import unzip

logger = getLogger(__name__)


class NLSTDataDownloader(BaseDataDownloader):
    """NLST downloader"""

    DATASET_URL = r"https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR/download/NLST2023.zip"  # pylint: disable=line-too-long

    def __init__(self) -> None:
        super().__init__(dataset_name="NLST")

    def _get_license_agreement_question(self) -> str:
        return (
            "NLST data is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://learn2reg.grand-challenge.org/Datasets/. "
            "(y/n) \n\n"
        )

    def _download_and_process(self, data_folder: str) -> None:
        dataset_zip_path = join(data_folder, "NLST2023.zip")
        logger.info("Downloading NLST dataset (progress might not be visible)...")
        download(
            source_url=self.DATASET_URL,
            target_path=dataset_zip_path,
            description="Downloading NLST dataset",
        )
        logger.info("Extracting NLST dataset")
        unzip(file_path=dataset_zip_path, extract_to_same_dir=True, remove_after=True)


class NLSTData(BaseVolumetricRegistrationData):
    """Class for accessing NLST cases"""

    def __init__(
        self,
        data_root: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            data_downloader=NLSTDataDownloader(),
        )
        with open(
            join(self._data_location, "NLST/NLST_dataset.json"), mode="r", encoding="utf-8"
        ) as dataset_info_file:
            dataset_info = json_load(dataset_info_file)
        self._validate_pairs = [
            (self._get_case_name(pair["moving"]), self._get_case_name(pair["fixed"]))
            for pair in dataset_info["registration_val"]
        ]
        self._train_cases = [
            self._get_case_name(pair["fixed"]) for pair in dataset_info["training_paired_images"]
        ]

    @staticmethod
    def _get_case_name(path: str) -> str:
        return "_".join(path.split("/")[-1].split("_")[:2])

    def _get_landmark_shift_in_voxel_coordinates(
        self,
        case_name: str,
        args: VolumetricDataArgs,
        registration_index: int,
    ) -> Sequence[float]:
        if args.crop is None:
            inital_shift: tuple[int, ...] = (0, 0, 0)
        else:
            inital_shift = tuple(-crop_left for (crop_left, _crop_right) in args.crop)
        if args.crop_or_pad_to is None:
            combined_cropping_shift = inital_shift
        else:
            output_shape_after_first_crop = self._get_output_shape_after_first_crop(
                case_name=case_name,
                args=args,
                registration_index=registration_index,
            )
            additional_shift = (
                int((abs(target - current) // 2) * copysign(1, target - current))
                for current, target in zip(output_shape_after_first_crop, args.crop_or_pad_to)
            )
            combined_cropping_shift = tuple(
                initial + additional for initial, additional in zip(inital_shift, additional_shift)
            )
        return combined_cropping_shift

    def get_case_landmarks(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        """Get landmarks for given case and registration index"""
        landmarks = from_numpy(
            loadtxt(
                join(
                    self._data_location,
                    "NLST",
                    "keypointsTr",
                    f"{case_name}_{registration_index:04d}.csv",
                ),
                delimiter=",",
            )
        ).to(get_default_dtype())
        landmark_shift = landmarks.new(
            self._get_landmark_shift_in_voxel_coordinates(
                case_name=case_name,
                args=args,
                registration_index=registration_index,
            )
        )
        return (landmarks + landmark_shift).transpose(0, 1)

    def get_case_affine(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        return from_numpy(self._get_spatial_image_for_case(case_name, registration_index).affine)

    def _get_raw_shape_for_case(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Sequence[int]:
        return self._get_spatial_image_for_case(case_name, registration_index).shape

    def _get_raw_data_for_case(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        mask = self._get_mask_spatial_image_for_case(case_name, registration_index).get_fdata()
        mask = from_numpy(mask).to(get_default_dtype())
        data = self._get_spatial_image_for_case(case_name, registration_index).get_fdata()
        return ((from_numpy(data).to(get_default_dtype()) + 1024.0) * mask)[None]

    def _get_raw_segmentation_for_case(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        raise NotImplementedError("Segmentation not available for Lung250M-4B")

    def _get_raw_mask_for_case(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        mask = self._get_mask_spatial_image_for_case(case_name, registration_index).get_fdata()
        mask = from_numpy(mask).to(get_default_dtype())
        return mask[None]

    def _get_path_to_case(self, case_name: str, registration_index: int) -> str:
        return join(
            self._data_location,
            "NLST",
            "imagesTr",
            f"{case_name}_{registration_index:04d}.nii.gz",
        )

    def _get_path_to_case_mask(self, case_name: str, registration_index: int) -> str:
        return join(
            self._data_location,
            "NLST",
            "masksTr",
            f"{case_name}_{registration_index:04d}.nii.gz",
        )

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return self._validate_pairs

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        raise NotImplementedError("Test data not available for NLST")

    def get_train_cases(self) -> Sequence[str]:
        return self._train_cases

    def _get_spatial_image_for_case(self, case_name: str, registration_index: int):
        return nib_load(self._get_path_to_case(case_name, registration_index))

    def _get_mask_spatial_image_for_case(self, case_name: str, registration_index: int):
        return nib_load(self._get_path_to_case_mask(case_name, registration_index))
