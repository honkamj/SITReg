"""Abdomen CT-CT dataset"""

from itertools import combinations, permutations
from logging import getLogger
from os import listdir, makedirs, remove
from os.path import isdir, join
from shutil import move, rmtree
from typing import Sequence, cast

from nibabel import Nifti1Image  # type: ignore
from nibabel import load as nib_load
from nibabel import save as nib_save
from numpy import abs as np_abs
from numpy import sum as np_sum
from torch import Tensor, from_numpy, get_default_dtype, ones, tensor

from algorithm.affine_transformation import embed_transformation, generate_scale_matrix
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.interpolator import LinearInterpolator
from data.base import BaseDataDownloader, BaseVolumetricRegistrationData
from data.interface import VolumetricDataArgs
from util.download import download
from util.extract import ungzip, unzip

from .air import get_source_voxel_size, inverse_transform_volume_by_air, read_air_file

logger = getLogger(__name__)


class LPBA40DataDownloader(BaseDataDownloader):
    """LPBA40 data downloader"""

    DELINEATION_DATA_URL = "https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Delineation_Space_MRI_and_label_files.zip"  # pylint: disable=line-too-long
    IMAGE_DATA_URL = "https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Native_Space_MRI_data_and_masks.zip"  # pylint: disable=line-too-long

    def __init__(self) -> None:
        super().__init__(dataset_name="lpba40")

    def _get_license_agreement_question(self) -> str:
        return (
            "LPBA40 dataset "
            "(Shattuck, David W., et al. Construction of a 3D probabilistic atlas "
            "of human cortical structures. Neuroimage 39.3 (2008): 1064-1080.) "
            "is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://resource.loni.usc.edu/resources/atlases-downloads/. "
            "(y/n) "
        )

    def _download_and_process(self, data_folder: str) -> None:
        delineation_data_zip_path = join(
            data_folder, "LPBA40_Subjects_Delineation_Space_MRI_and_label_files.zip"
        )
        image_data_zip_path = join(
            data_folder, "LPBA40_Subjects_Native_Space_MRI_data_and_masks.zip"
        )
        logger.info("Downloading LPBA40 dataset")
        download(
            source_url=self.DELINEATION_DATA_URL,
            target_path=delineation_data_zip_path,
            description="Downloading LPBA40 delineation space data",
        )
        download(
            source_url=self.IMAGE_DATA_URL,
            target_path=image_data_zip_path,
            description="Downloading LPBA40 native space data",
        )
        logger.info("Reorganizing data...")
        unzip(file_path=delineation_data_zip_path, extract_to_same_dir=True, remove_after=True)
        unzip(
            file_path=join(data_folder, "LPBA40subjects.delineation_space.zip"),
            extract_to_same_dir=True,
            remove_after=True,
        )
        unzipped_delineation_folder = join(data_folder, "LPBA40", "delineation_space")
        move(
            join(unzipped_delineation_folder, "lpba40.label.xml"),
            join(data_folder, "lpba40.label.xml"),
        )
        for case_name in listdir(unzipped_delineation_folder):
            case_folder = join(unzipped_delineation_folder, case_name)
            if isdir(case_folder):
                target_case_folder = join(data_folder, case_name)
                makedirs(target_case_folder)
                move(
                    join(case_folder, f"{case_name}.delineation.structure.label.hdr"),
                    join(target_case_folder, f"{case_name}.delineation.structure.label.hdr"),
                )
                move(
                    join(case_folder, f"{case_name}.delineation.structure.label.img.gz"),
                    join(target_case_folder, f"{case_name}.delineation.structure.label.img.gz"),
                )
        unzipped_root_folder = join(data_folder, "LPBA40")
        rmtree(join(unzipped_root_folder, "delineation_space"))
        for licence_file in listdir(unzipped_root_folder):
            move(
                join(unzipped_root_folder, licence_file),
                join(data_folder, licence_file),
            )
        rmtree(unzipped_root_folder)
        unzip(file_path=image_data_zip_path, extract_to_same_dir=True, remove_after=True)
        unzip(
            file_path=join(data_folder, "LPBA40subjects.native_space.zip"),
            extract_to_same_dir=True,
            remove_after=True,
        )
        unzipped_image_folder = join(data_folder, "LPBA40", "native_space")
        for case_name in listdir(unzipped_image_folder):
            case_folder = join(unzipped_image_folder, case_name)
            if isdir(case_folder):
                target_case_folder = join(data_folder, case_name)
                move(
                    join(case_folder, f"{case_name}.native.brain.bfc.hdr"),
                    join(target_case_folder, f"{case_name}.native.brain.bfc.hdr"),
                )
                move(
                    join(case_folder, f"{case_name}.native.brain.bfc.img.gz"),
                    join(target_case_folder, f"{case_name}.native.brain.bfc.img.gz"),
                )
                move(
                    join(case_folder, "tissue", f"{case_name}.native.tissue.hdr"),
                    join(target_case_folder, f"{case_name}.native.tissue.hdr"),
                )
                move(
                    join(case_folder, "tissue", f"{case_name}.native.tissue.img.gz"),
                    join(target_case_folder, f"{case_name}.native.tissue.img.gz"),
                )
                move(
                    join(case_folder, "transforms", f"{case_name}.delineation.to.native.air"),
                    join(target_case_folder, f"{case_name}.delineation.to.native.air"),
                )
        rmtree(unzipped_root_folder)
        logger.info("Processing data...")
        for case_name in listdir(data_folder):
            case_folder = join(data_folder, case_name)
            if isdir(case_folder):
                ungzipped_names = (
                    f"{case_name}.delineation.structure.label.img.gz",
                    f"{case_name}.native.brain.bfc.img.gz",
                    f"{case_name}.native.tissue.img.gz",
                )
                for ungzipped_name in ungzipped_names:
                    ungzip(
                        join(case_folder, ungzipped_name),
                        remove_after=True,
                    )
                self._transform_analyze_by_air_and_save_to_nifti(
                    case_folder=case_folder,
                    source_header_name=f"{case_name}.native.brain.bfc.hdr",
                    target_name=f"{case_name}.delineation.brain.bfc.nii.gz",
                    air_name=f"{case_name}.delineation.to.native.air",
                )
                remove(join(case_folder, f"{case_name}.native.brain.bfc.hdr"))
                remove(join(case_folder, f"{case_name}.native.brain.bfc.img"))
                self._transform_analyze_by_air_and_save_to_nifti(
                    case_folder=case_folder,
                    source_header_name=f"{case_name}.native.tissue.hdr",
                    target_name=f"{case_name}.delineation.tissue.nii.gz",
                    air_name=f"{case_name}.delineation.to.native.air",
                )
                remove(join(case_folder, f"{case_name}.native.tissue.hdr"))
                remove(join(case_folder, f"{case_name}.native.tissue.img"))
                self._convert_analyze_to_nifti(
                    case_folder=case_folder,
                    source_header_name=f"{case_name}.delineation.structure.label.hdr",
                    target_name=f"{case_name}.delineation.structure.label.nii.gz",
                )
                remove(join(case_folder, f"{case_name}.delineation.structure.label.hdr"))
                remove(join(case_folder, f"{case_name}.delineation.structure.label.img"))
                self._white_matter_normalize(
                    case_folder=case_folder,
                    normalized_name=f"{case_name}.delineation.brain.bfc.nii.gz",
                    tissue_mask_name=f"{case_name}.delineation.tissue.nii.gz",
                )
                self._apply_mask_to_label(
                    case_folder=case_folder,
                    label_name=f"{case_name}.delineation.brain.bfc.nii.gz",
                    tissue_mask_name=f"{case_name}.delineation.tissue.nii.gz",
                )
                remove(join(case_folder, f"{case_name}.delineation.tissue.nii.gz"))
                remove(join(case_folder, f"{case_name}.delineation.to.native.air"))

    @staticmethod
    def _swap_voxel_size_for_two_dims(voxel_size: Tensor) -> Tensor:
        swapped_voxel_size = voxel_size.clone()
        swapped_voxel_size[0] = -voxel_size[0]
        swapped_voxel_size[1] = -voxel_size[1]
        return swapped_voxel_size

    @classmethod
    def _transform_analyze_by_air_and_save_to_nifti(
        cls, case_folder: str, source_header_name: str, target_name: str, air_name: str
    ) -> None:
        image = cast(Nifti1Image, nib_load(join(case_folder, source_header_name)))
        image_tensor = from_numpy(image.get_fdata()[None, ..., 0])
        with open(join(case_folder, air_name), "rb") as air_file:
            air = read_air_file(air_file)
        voxel_size = image.header["pixdim"][1:4]
        transformed_image = inverse_transform_volume_by_air(
            air=air,
            volume=image_tensor,
            target_voxel_size=voxel_size,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=False
            ),
        )
        nifti_image = Nifti1Image(
            transformed_image.numpy()[0],
            affine=embed_transformation(
                generate_scale_matrix(
                    cls._swap_voxel_size_for_two_dims(tensor(get_source_voxel_size(air)))
                ),
                (4, 4),
            ).numpy(),
        )
        nib_save(nifti_image, join(case_folder, target_name))

    @classmethod
    def _convert_analyze_to_nifti(
        cls,
        case_folder: str,
        source_header_name: str,
        target_name: str,
    ) -> None:
        image = cast(Nifti1Image, nib_load(join(case_folder, source_header_name)))
        voxel_size = tensor(image.header["pixdim"][1:4])
        nifti_image = Nifti1Image(
            image.get_fdata()[..., 0],
            affine=embed_transformation(
                generate_scale_matrix(cls._swap_voxel_size_for_two_dims(voxel_size)),
                (4, 4),
            ).numpy(),
        )
        nib_save(nifti_image, join(case_folder, target_name))

    @classmethod
    def _white_matter_normalize(
        cls,
        case_folder: str,
        normalized_name: str,
        tissue_mask_name: str,
    ) -> None:
        image = cast(Nifti1Image, nib_load(join(case_folder, normalized_name)))
        tissue_mask_image = cast(Nifti1Image, nib_load(join(case_folder, tissue_mask_name)))
        wm_tissue_mask = np_abs(tissue_mask_image.get_fdata() - 3) < 1e-5
        image_data = image.get_fdata()
        normalized_image = image_data / (
            np_sum(image_data * wm_tissue_mask) / np_sum(wm_tissue_mask)
        )
        nifti_image = Nifti1Image(
            normalized_image,
            affine=image.affine,
        )
        nib_save(nifti_image, join(case_folder, normalized_name))

    @classmethod
    def _apply_mask_to_label(
        cls,
        case_folder: str,
        label_name: str,
        tissue_mask_name: str,
    ) -> None:
        image = cast(Nifti1Image, nib_load(join(case_folder, label_name)))
        tissue_mask_image = cast(Nifti1Image, nib_load(join(case_folder, tissue_mask_name)))
        brain_mask = tissue_mask_image.get_fdata() > 0
        image_data = image.dataobj[...]
        masked_image = (image_data * brain_mask).astype(image_data.dtype)
        nifti_image = Nifti1Image(
            masked_image,
            affine=image.affine,
        )
        nib_save(nifti_image, join(case_folder, label_name))


class LPBA40Data(BaseVolumetricRegistrationData):
    """Class for accessing LPBA40 cases"""

    VALIDATE_CASES = ["S14", "S16", "S19", "S26", "S39"]
    TEST_CASES = ["S11", "S12", "S13", "S15", "S24", "S28", "S29", "S32", "S34", "S38"]
    TRAIN_CASES = [
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S09",
        "S10",
        "S17",
        "S18",
        "S20",
        "S21",
        "S22",
        "S23",
        "S25",
        "S27",
        "S30",
        "S31",
        "S33",
        "S35",
        "S36",
        "S37",
        "S40",
    ]

    def __init__(self, data_root: str, both_directions: bool) -> None:
        super().__init__(data_root, LPBA40DataDownloader())
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
        return join(self._data_location, case_name, f"{case_name}.{file_type}.nii.gz")

    def _get_pairs(self, cases: Sequence[str]) -> list[tuple[str, str]]:
        pair_iterator = permutations if self._both_directions else combinations
        return cast(list[tuple[str, str]], list(pair_iterator(cases, r=2)))

    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        return self._get_pairs(self.VALIDATE_CASES)

    def _get_test_pairs(self) -> list[tuple[str, str]]:
        return self._get_pairs(self.TEST_CASES)

    def get_train_cases(self) -> Sequence[str]:
        return self.TRAIN_CASES

    def _get_cases(self) -> list[str]:
        return listdir(self._data_location)

    def _get_spatial_image_for_case(self, case_name: str, file_type: str):
        return nib_load(self._get_path_to_case(case_name, file_type))
