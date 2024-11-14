"""Lung250M-4B dataset"""

from logging import getLogger
from math import copysign
from os import listdir
from os.path import join
from typing import Sequence

from nibabel import load as nib_load  # type: ignore
from torch import Tensor, from_numpy, get_default_dtype, load

from data.base import BaseDataDownloader, BaseVolumetricRegistrationData
from data.interface import VolumetricDataArgs
from util.download import download
from util.extract import unzip

logger = getLogger(__name__)


class Lung250M4BDataDownloader(BaseDataDownloader):
    """Lung250M-4B downloader"""

    DATASET_URL = r"https://cloud.imi.uni-luebeck.de/s/s64fqbPpXNexBPP/download?path=%2F&files=%5B%22masksTs%22%2C%22masksTr%22%2C%22imagesTr%22%2C%22imagesTs%22%2C%22segTr%22%2C%22segTs%22%2C%22lms_validation.pth%22%5D"  # pylint: disable=line-too-long

    def __init__(self) -> None:
        super().__init__(dataset_name="Lung250M-4B")

    def _get_license_agreement_question(self) -> str:
        return (
            "Lung250M-4B data is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://github.com/multimodallearning/Lung250M-4B. "
            "(y/n) \n\n"
            "Note also that for Lung250M-4B dataset some of the cases (Dirlab-COPD) used "
            "in the SITReg paper have to be downloaded manually from "
            "https://github.com/multimodallearning/Lung250M-4B and that the images from "
            "EMPIRE data set which are also part of Lung250M-4B and require manual "
            "downloading were not used in the SITReg paper due to their terms of use. "
        )

    def _download_and_process(self, data_folder: str) -> None:
        dataset_zip_path = join(data_folder, "Lung250M-4B.zip")
        logger.info("Downloading Lung250M-4B dataset (progress might not be visible)...")
        download(
            source_url=self.DATASET_URL,
            target_path=dataset_zip_path,
            description="Downloading Lung250M-4B dataset",
        )
        logger.info("Extracting Lung250M-4B dataset")
        unzip(file_path=dataset_zip_path, extract_to_same_dir=True, remove_after=True)


class Lung250M4BData(BaseVolumetricRegistrationData):
    """Class for accessing Lung250M-4B cases"""

    VALIDATE_CASES = [
        # "case_002", # Part of EMPIRE10 dataset which we can not use due to terms of use
        # "case_008", # Part of EMPIRE10 dataset which we can not use due to terms of use
        "case_054",
        "case_055",
        "case_056",
        "case_094",
        "case_097",
    ] + [f"case_{i:03d}" for i in range(113, 124)]
    TEST_CASES = [f"case_{i:03d}" for i in range(104, 113)]

    def __init__(
        self,
        data_root: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            data_downloader=Lung250M4BDataDownloader(),
        )

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
        final_shift = tuple(shift - 1 / 2 for shift in combined_cropping_shift)
        return final_shift

    def get_case_landmarks(
        self, case_name: str, args: VolumetricDataArgs, registration_index: int
    ) -> Tensor:
        """Get landmarks for given case and registration index"""
        both_landmarks: Tensor = load(join(self._data_location, "lms_validation.pth"))[
            str(int(case_name[-3:]))
        ]
        if registration_index == 0:
            landmarks = both_landmarks[:, :3]
        else:
            landmarks = both_landmarks[:, 3:]
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
        case_folder = (
            "imagesTs" if case_name in self.TEST_CASES + self.VALIDATE_CASES else "imagesTr"
        )
        return join(
            self._data_location,
            case_folder,
            f"{case_name}_{registration_index + 1}.nii.gz",
        )

    def _get_path_to_case_mask(self, case_name: str, registration_index: int) -> str:
        case_folder = "masksTs" if case_name in self.TEST_CASES + self.VALIDATE_CASES else "masksTr"
        return join(
            self._data_location,
            case_folder,
            f"{case_name}_{registration_index + 1}.nii.gz",
        )

    def get_inference_pairs(self, division: str) -> Sequence[tuple[str, str]]:
        if division == "train":
            return [(case, case) for case in self.get_train_cases()]
        return super().get_inference_pairs(division)

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return [(case, case) for case in self.VALIDATE_CASES]

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        return [(case, case) for case in self.TEST_CASES]

    def get_train_cases(self) -> Sequence[str]:
        return self._get_cases("imagesTr")

    def _get_cases(self, subfolder: str) -> list[str]:
        raw_names = listdir(join(self._data_location, subfolder))
        case_names = sorted(set(name[:-9] for name in raw_names if name.endswith(".nii.gz")))
        return case_names

    def _get_spatial_image_for_case(self, case_name: str, registration_index: int):
        return nib_load(self._get_path_to_case(case_name, registration_index))

    def _get_mask_spatial_image_for_case(self, case_name: str, registration_index: int):
        return nib_load(self._get_path_to_case_mask(case_name, registration_index))
