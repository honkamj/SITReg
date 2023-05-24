"""Base data classes"""

from abc import abstractmethod
from datetime import datetime
from itertools import product
from logging import getLogger
from multiprocessing import Value
from os import environ, listdir, makedirs
from os.path import isdir, isfile, join
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError

from numpy import ones as np_ones
from numpy.ma import masked_invalid as np_masked_invalid
from surface_distance import compute_robust_hausdorff, compute_surface_distances  # type: ignore
from torch import Generator, Tensor, as_tensor
from torch import bool as torch_bool
from torch import zeros
from torch.nn.functional import avg_pool3d
from torch.utils.data import DataLoader, Dataset

from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs, as_displacement_field
from algorithm.dense_deformation import generate_voxel_coordinate_grid
from algorithm.everywhere_differentiable_determinant import calculate_determinant
from algorithm.finite_difference import estimate_spatial_jacobian_matrices
from algorithm.interpolator import LinearInterpolator, NearestInterpolator
from data.interface import (
    IDataDownloader,
    IEvaluator,
    IInferenceFactory,
    InferenceMetadata,
    IStorageFactory,
    IVolumetricRegistrationData,
    IVolumetricRegistrationInferenceDataset,
    VolumetricDataArgs,
)
from util.metrics import ISummarizer, LastSummarizer, MeanSummarizer, StdSummarizer


logger = getLogger()


class BaseVariantDataset(Dataset):
    """Base dataset which supports generating variants with multiprocessing"""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._random_generator: Generator | None = None
        self._shared_generation = Value("i", 0)
        self._local_generation = -1

    @abstractmethod
    def _generate_new_variant(self, random_generator: Generator, generation: int) -> None:
        """Generate new variant of the dataset

        This is called always at least once when fetching the first item
        or dataset length.
        """

    @abstractmethod
    def _length(self, generation: int) -> int:
        """Return length of the dataset"""

    @abstractmethod
    def _get_item(self, index: int, generation: int) -> Any:
        """Return item"""

    def generate_new_variant(self) -> None:
        """Generate new variant of the dataset"""
        with self._shared_generation.get_lock():
            self._shared_generation.value += 1  # type: ignore

    def __len__(self) -> int:
        return self._length(self._shared_generation.value)  # type: ignore

    def __getitem__(self, index: int) -> Any:
        self._sync_generation()
        if index >= self._length(self._local_generation):
            raise IndexError("Index out of range")
        return self._get_item(index, self._local_generation)

    def _sync_generation(self) -> None:
        while self._local_generation < self._shared_generation.value:  # type: ignore
            self._local_generation += 1
            # Postpone generating the random generator after the process spawing since
            # it can not be pickled.
            if self._random_generator is None:
                self._random_generator = Generator().manual_seed(self._seed)
            self._generate_new_variant(self._random_generator, self._local_generation)


class BaseDataDownloader(IDataDownloader):
    """Base data downloader class"""

    def __init__(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

    @abstractmethod
    def _download_and_process(self, data_folder: str) -> None:
        """Download the data to data_folder and process the data"""

    @abstractmethod
    def _get_license_agreement_question(self) -> str:
        """Provide question asked before allowing to download the data"""

    def download(self, data_root: str) -> str:
        """Download the dataset to data_root

        Returns: Path to the data folder
        """
        data_folder = self._get_data_folder(data_root)
        if not self._is_downloaded(data_folder):
            if environ.get("AGREE_TO_DATA_TERMS_OF_USE_cuRfC7gemUBKVGZv91ey", "").lower() != "true":
                answer = input(self._get_license_agreement_question())
                if answer != "y":
                    raise RuntimeError(
                        "Can not download the dataset without agreeing to the terms of use!"
                    )
            self._ensure_target_folder_empty(data_folder)
            self._create_data_folder(data_folder)
            try:
                self._download_and_process(data_folder)
            except HTTPError:
                logger.error("Downloading the data failed")
                raise
            self._write_timestamp(data_folder)
        return data_folder

    def _get_data_folder(self, data_root: str) -> str:
        return join(data_root, self._dataset_name)

    @staticmethod
    def _is_downloaded(data_folder: str) -> bool:
        return isfile(join(data_folder, "timestamp.txt"))

    @staticmethod
    def _ensure_target_folder_empty(data_folder: str) -> None:
        if isdir(data_folder):
            if len(listdir(data_folder)) != 0:
                raise RuntimeError(f"Target directory {data_folder} is not empty.")

    @staticmethod
    def _create_data_folder(data_folder: str) -> None:
        makedirs(data_folder, exist_ok=True)

    @staticmethod
    def _write_timestamp(data_folder: str) -> None:
        with open(join(data_folder, "timestamp.txt"), mode="w", encoding="utf-8") as timestamp_file:
            timestamp_file.write(str(datetime.now()))


class SequenceDataset(Dataset):
    """Dataset based on sequence"""

    def __init__(self, sequence: Sequence[Any]) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index: int) -> Any:
        return self._sequence[index]


class BaseEvaluator(IEvaluator):
    """Base evaluator implementation"""

    def __call__(self, inference_outputs: Mapping[str, Any]) -> Mapping[str, float]:
        return {
            "inference_time": inference_outputs["inference_time"],
            "inference_memory_usage": inference_outputs["inference_memory_usage"],
            "device_name": inference_outputs["device_name"],
        }

    @property
    def evaluation_inference_outputs(self) -> set[str]:
        return {"inference_time", "inference_memory_usage", "device_name"}


class BaseInferenceFactory(IInferenceFactory):
    """Base inference factory"""

    def get_evaluator_summarizers(self) -> Mapping[str | None, Iterable[ISummarizer]]:
        return {None: (MeanSummarizer(), StdSummarizer()), "device_name": (LastSummarizer(),)}


class BaseVolumetricRegistrationData(IVolumetricRegistrationData):
    """Class for accessing volumetric subject-to-subject registration data"""

    def __init__(self, data_root: str, data_downloader: IDataDownloader) -> None:
        self._data_location = data_downloader.download(data_root)

    def get_case_shape(self, case_name: str, args: VolumetricDataArgs) -> Sequence[int]:
        """Get shape for the given case"""
        shape = self._get_raw_shape_for_case(case_name, args)
        if args.downsampling_factor is None:
            downsampling_factor: Sequence[int] = [1] * len(shape)
        else:
            downsampling_factor = args.downsampling_factor
        if args.crop is None:
            crop: Sequence[tuple[int, int]] = [(0, 0)] * len(shape)
        else:
            crop = args.crop
        return [
            dim_size // dim_downsampling_factor - crop_left - crop_right
            for dim_size, dim_downsampling_factor, (crop_left, crop_right) in zip(
                shape, downsampling_factor, crop
            )
        ]

    @abstractmethod
    def get_case_affine(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        """Get affine transformation for the given case"""

    def get_case_volume(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        """Get volume for the given case"""
        return self._modify_volume(
            self._get_raw_data_for_case(case_name, args),
            modifiers=[
                self._downsample,
                self._crop,
                self._clip,
                self._shift_and_normalize,
                self._normalize,
            ],
            args=args,
        )

    def get_case_mask(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        return self._modify_volume(
            self._get_raw_mask_for_case(case_name, args),
            modifiers=[
                self._downsample,
                self._crop,
                self._threshold_mask,
            ],
            args=args,
        )

    @staticmethod
    def _modify_volume(
        volume: Tensor,
        modifiers: list[Callable[[Tensor, VolumetricDataArgs], Tensor]],
        args: VolumetricDataArgs,
    ) -> Tensor:
        modified = volume
        for modifier in modifiers:
            modified = modifier(modified, args)
        return modified

    @abstractmethod
    def _get_raw_shape_for_case(self, case_name: str, args: VolumetricDataArgs) -> Sequence[int]:
        """Get case shape before any modifications"""

    @abstractmethod
    def _get_raw_mask_for_case(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        """Get raw mask for case"""

    @abstractmethod
    def _get_raw_data_for_case(self, case_name: str, args: VolumetricDataArgs) -> Tensor:
        """Get raw data for case"""

    @staticmethod
    def _downsample(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if args.downsampling_factor is None or all(
            factor == 1 for factor in args.downsampling_factor
        ):
            return image
        downsampled = avg_pool3d(
            input=image[None, None],
            kernel_size=tuple(args.downsampling_factor),
            stride=tuple(args.downsampling_factor),
        )[0, 0]
        if "seg" in args.file_type:
            downsampled = downsampled.round()
        return downsampled

    @staticmethod
    def _crop(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if args.crop is None:
            return image
        crop_slice = tuple(
            slice(crop_left, -crop_right) if crop_right != 0 else slice(crop_left, None)
            for crop_left, crop_right in args.crop
        )
        return image[crop_slice].clone()

    @staticmethod
    def _normalize(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if not args.normalize:
            return image
        image_max = image.max()
        image_min = image.min()
        normalized = (image - image_min) / (image_max - image_min)
        return normalized

    @staticmethod
    def _clip(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if args.clip is None:
            return image
        return image.clamp(min=args.clip[0], max=args.clip[1])

    @staticmethod
    def _shift_and_normalize(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if args.shift_and_normalize is None:
            return image
        return (image - args.shift_and_normalize[0]) / args.shift_and_normalize[1]

    @staticmethod
    def _threshold_mask(image: Tensor, args: VolumetricDataArgs) -> Tensor:
        if args.mask_threshold is None:
            return image
        return (image < args.mask_threshold).logical_not().type(image.dtype)

    def get_inference_pairs(self, division: str) -> Sequence[tuple[str, str]]:
        """Get inference pairs"""
        if division == "validate":
            return self._get_validate_pairs()
        elif division == "test":
            return self._get_test_pairs()
        raise ValueError(f"Unknown division {division}")

    @abstractmethod
    def _get_validate_pairs(self) -> list[tuple[str, str]]:
        """Get validation pairs"""

    @abstractmethod
    def _get_test_pairs(self) -> list[tuple[str, str]]:
        """Get test pairs"""

    @abstractmethod
    def get_train_cases(self) -> Sequence[str]:
        """Get training cases"""

    @abstractmethod
    def _get_cases(self) -> list[str]:
        """Get all cases"""


class BaseVolumetricRegistrationInferenceFactory(BaseInferenceFactory):
    """Oasis inference factory"""

    def __init__(self, dataset: IVolumetricRegistrationInferenceDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    @abstractmethod
    def _get_storage_factory(self, affine: Tensor) -> IStorageFactory:
        """Get storage factory with affine"""

    def get_metadata(self, index: int) -> InferenceMetadata:
        image_1_name, image_2_name = self._dataset.names(index)
        image_1_shape, image_2_shape = self._dataset.shapes(index)
        image_1_affine, image_2_affine = self._dataset.affines(index)
        return InferenceMetadata(
            inference_name=f"{image_1_name}-{image_2_name}",
            names=[image_1_name, image_2_name],
            info={
                "image_1_shape": image_1_shape,
                "image_2_shape": image_2_shape,
                "image_1_affine": image_1_affine,
                "image_2_affine": image_2_affine,
            },
            default_storage_factories=[
                self._get_storage_factory(image_1_affine),
                self._get_storage_factory(image_2_affine),
            ],
        )

    def get_data_loader(self, index: int, num_workers: int) -> DataLoader:
        if num_workers > 0:
            raise RuntimeError("No multiprocessing suppport (nor need).")
        return DataLoader(dataset=SequenceDataset([self._dataset[index]]))

    def generate_dummy_batch_and_metadata(self) -> tuple[tuple[Tensor, Tensor], InferenceMetadata]:
        image_1_shape, image_2_shape = self._dataset.shapes(0)
        image_1_affine, image_2_affine = self._dataset.affines(0)
        batch = (zeros((1, 1) + tuple(image_1_shape)), zeros((1, 1) + tuple(image_2_shape)))
        return (
            batch,
            InferenceMetadata(
                inference_name="dummy_batch",
                names=["image_1", "image_2"],
                info={
                    "image_1_shape": image_1_shape,
                    "image_2_shape": image_2_shape,
                    "image_1_affine": image_1_affine,
                    "image_2_affine": image_2_affine,
                },
                default_storage_factories=[
                    self._get_storage_factory(image_1_affine),
                    self._get_storage_factory(image_2_affine),
                ],
            ),
        )


class BaseVolumetricRegistrationSegmentationEvaluator(BaseEvaluator):
    """Base volumetric registration segmentation evaluator"""

    def __init__(
        self,
        source_mask_seg: Tensor,
        target_mask_seg: Tensor,
        metrics_to_compute: Sequence[str],
        source_temp_storage_factory: IStorageFactory,
        source_name: str,
        target_temp_storage_factory: IStorageFactory,
        target_name: str,
    ) -> None:
        super().__init__()
        self._metrics_to_compute = metrics_to_compute
        self._source_mask_seg = source_mask_seg
        self._target_mask_seg = target_mask_seg
        self._source_temp_storage_factory = source_temp_storage_factory
        self._source_name = source_name
        self._target_temp_storage_factory = target_temp_storage_factory
        self._target_name = target_name

    @property
    @abstractmethod
    def _names_to_indices_seg(self) -> Mapping[str, Sequence[int]]:
        """Get names to indices of the segmentation mask"""

    def __call__(self, inference_outputs: Mapping[str, Any]) -> Mapping[str, float]:
        metrics: dict[str, int | float] = {}
        evaluation_temp_folder = environ.get("EVALUATION_TEMP_FOLDER")
        if evaluation_temp_folder is not None:
            source_seg_storage = self._source_temp_storage_factory.create(
                f"{self._source_name}_seg"
            )
            target_seg_storage = self._source_temp_storage_factory.create(
                f"{self._target_name}_seg"
            )
            source_seg_storage.save(self._source_mask_seg, evaluation_temp_folder)
            target_seg_storage.save(self._target_mask_seg, evaluation_temp_folder)
        for index, forward_displacement_field in enumerate(
            inference_outputs["forward_displacement_field"]
        ):
            if forward_displacement_field is not None:
                transformed_source_mask_seg = self._transform_mask(
                    mask=self._source_mask_seg,
                    displacement_field=forward_displacement_field,
                )
                if evaluation_temp_folder is not None:
                    forward_seg_resampled_storage = self._source_temp_storage_factory.create(
                        f"{self._source_name}-{self._target_name}_seg_resampled"
                    )
                    forward_seg_resampled_storage.save(
                        transformed_source_mask_seg, evaluation_temp_folder
                    )

                metrics.update(
                    self._compute_segmentation_metrics(
                        mask_1_seg=transformed_source_mask_seg,
                        mask_2_seg=self._target_mask_seg,
                        prefix=f"input_order_{index}_forward_",
                    )
                )
                metrics.update(
                    self._compute_determinant_metrics(
                        displacement_field=forward_displacement_field,
                        prefix=f"input_order_{index}_forward_",
                    )
                )
        for index, inverse_displacement_field in enumerate(
            inference_outputs["inverse_displacement_field"]
        ):
            if inverse_displacement_field is not None:
                transformed_target_mask_seg = self._transform_mask(
                    mask=self._target_mask_seg,
                    displacement_field=inverse_displacement_field,
                )
                if evaluation_temp_folder is not None:
                    inverse_seg_resampled_storage = self._target_temp_storage_factory.create(
                        f"{self._target_name}-{self._source_name}_seg_resampled"
                    )
                    inverse_seg_resampled_storage.save(
                        transformed_target_mask_seg, evaluation_temp_folder
                    )
                metrics.update(
                    self._compute_segmentation_metrics(
                        mask_1_seg=transformed_target_mask_seg,
                        mask_2_seg=self._source_mask_seg,
                        prefix=f"input_order_{index}_inverse_",
                    )
                )
                metrics.update(
                    self._compute_determinant_metrics(
                        displacement_field=inverse_displacement_field,
                        prefix=f"input_order_{index}_inverse_",
                    )
                )
        for index_0, index_1 in product(
            range(len(inference_outputs["forward_displacement_field"])),
            range(len(inference_outputs["inverse_displacement_field"])),
        ):
            if (
                inference_outputs["forward_displacement_field"][index_0] is not None
                and inference_outputs["inverse_displacement_field"][index_1] is not None
            ):
                metrics.update(
                    self._compute_inverse_consistency_metrics(
                        forward_displacement_field=inference_outputs["forward_displacement_field"][
                            index_0
                        ],
                        inverse_displacement_field=inference_outputs["inverse_displacement_field"][
                            index_1
                        ],
                        prefix=f"input_orders_{index_0}_{index_1}_",
                    )
                )
                metrics.update(
                    self._compute_inverse_consistency_metrics(
                        forward_displacement_field=inference_outputs["inverse_displacement_field"][
                            index_1
                        ],
                        inverse_displacement_field=inference_outputs["forward_displacement_field"][
                            index_0
                        ],
                        prefix=f"input_orders_{index_0}_{index_1}_reverse_",
                    )
                )
        return super().__call__(inference_outputs) | metrics

    def _compute_segmentation_metrics(
        self,
        mask_1_seg: Tensor,
        mask_2_seg: Tensor,
        prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if "dice" in self._metrics_to_compute or "hd95" in self._metrics_to_compute:
            seg_dice_metrics, seg_hd95_metrics = self._organ_segmentation_metrics(
                organs_mask_1=mask_1_seg,
                organs_mask_2=mask_2_seg,
                names_to_indices=self._names_to_indices_seg,
                prefix=prefix,
            )
            if "dice" in self._metrics_to_compute:
                metrics[f"{prefix}dice"] = float(
                    np_masked_invalid(list(seg_dice_metrics.values())).mean()
                )
            if "hd95" in self._metrics_to_compute:
                metrics[f"{prefix}hd95"] = float(
                    np_masked_invalid(list(seg_hd95_metrics.values())).mean()
                )
            metrics.update(seg_dice_metrics)
            metrics.update(seg_hd95_metrics)
        return metrics

    def _transform_mask(self, mask: Tensor, displacement_field: Tensor) -> Tensor:
        mask = mask[None]
        displacement_field = displacement_field[None]
        coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=mask.shape[2:], voxel_size=(1.0, 1.0, 1.0)
        )
        mask_mapping = ComposableFactory.create_volume(
            data=mask,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=NearestInterpolator(padding_mode="zeros"), mask_outside_fov=True
            ),
        )
        displacement_field_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=displacement_field,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=False
            ),
        )
        transformed_mask_masked = mask_mapping.compose(displacement_field_mapping)(
            coordinate_system.grid
        )
        transformed_mask = transformed_mask_masked.generate_values().round()
        transformed_mask_mask = transformed_mask_masked.generate_mask().bool()
        masked_transformed_mask = transformed_mask[0] * transformed_mask_mask[0]
        return masked_transformed_mask

    def _organ_segmentation_metrics(
        self,
        organs_mask_1: Tensor,
        organs_mask_2: Tensor,
        names_to_indices: Mapping[str, Sequence[int]],
        prefix: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        dice_metrics: dict[str, float] = {}
        hd95_metrics: dict[str, float] = {}
        for name, indices in names_to_indices.items():
            organ_mask_1 = zeros(organs_mask_1.shape, dtype=torch_bool, device=organs_mask_1.device)
            organ_mask_2 = zeros(organs_mask_2.shape, dtype=torch_bool, device=organs_mask_2.device)
            for index in indices:
                organ_mask_1 |= organs_mask_1 == index
                organ_mask_2 |= organs_mask_2 == index
            if "dice" in self._metrics_to_compute:
                organ_dice = self._dice(organ_mask_1, organ_mask_2)
                dice_metrics[f"{prefix}{name}_dice"] = organ_dice
            if "hd95" in self._metrics_to_compute:
                organ_hd95 = compute_robust_hausdorff(
                    compute_surface_distances(
                        organ_mask_2[0].cpu().numpy(),
                        organ_mask_1[0].cpu().numpy(),
                        np_ones(3),
                    ),
                    95.0,
                )
                hd95_metrics[f"{prefix}{name}_hd95"] = organ_hd95
        return dice_metrics, hd95_metrics

    @staticmethod
    def _dice(mask_1: Tensor, mask_2: Tensor) -> float:
        intersection = (mask_1 & mask_2).sum()
        union = mask_1.sum() + mask_2.sum()
        return (2 * intersection / union).item()

    def _compute_inverse_consistency_metrics(
        self,
        forward_displacement_field: Tensor,
        inverse_displacement_field: Tensor,
        prefix: str,
    ) -> dict[str, int | float]:
        if "inverse_consistency" not in self._metrics_to_compute:
            return {}
        forward_displacement_field = forward_displacement_field[None]
        inverse_displacement_field = inverse_displacement_field[None]
        coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=forward_displacement_field.shape[2:], voxel_size=(1.0, 1.0, 1.0)
        )
        forward_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=forward_displacement_field,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=True
            ),
        )
        inverse_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=inverse_displacement_field,
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=True
            ),
        )
        forward_composition_ddf, forward_composition_mask = as_displacement_field(
            forward_mapping.compose(inverse_mapping), coordinate_system=coordinate_system
        )
        assert forward_composition_mask is not None
        forward_composition_n_voxels = forward_composition_mask.sum()
        forward_composition_ddf_masked = forward_composition_ddf * forward_composition_mask
        return {
            f"{prefix}inverse_consistency_mse": forward_composition_ddf.square().mean().item(),
            f"{prefix}inverse_consistency_mse_masked": (
                forward_composition_ddf_masked.square().sum() / (3 * forward_composition_n_voxels)
            ).item(),
            f"{prefix}inverse_consistency_mae": forward_composition_ddf.abs().mean().item(),
            f"{prefix}inverse_consistency_mae_masked": (
                forward_composition_ddf_masked.abs().sum() / (3 * forward_composition_n_voxels)
            ).item(),
            f"{prefix}inverse_consistency_max": forward_composition_ddf.abs().max().item(),
            f"{prefix}inverse_consistency_max_masked": forward_composition_ddf_masked.abs()
            .max()
            .item(),
        }

    def _compute_determinant_metrics(
        self, displacement_field: Tensor, prefix: str
    ) -> dict[str, int | float]:
        if "determinant" not in self._metrics_to_compute:
            return {}
        n_negative_determinants_avg, det_std_avg = self._determinant_metrics(
            displacement_field, other_dims="average", central=False
        )
        n_negative_determinants_crop_last, det_std_crop_last = self._determinant_metrics(
            displacement_field, other_dims="crop_last", central=False
        )
        n_negative_determinants_central, det_std_central = self._determinant_metrics(
            displacement_field, other_dims="crop", central=True
        )
        n_voxels = int(
            as_tensor(displacement_field.shape[1:], device=displacement_field.device).prod()
        )
        return {
            f"{prefix}n_negative_determinants_avg_along_other_dims": n_negative_determinants_avg,
            f"{prefix}proportion_negative_determinants_avg_along_other_dims": (
                n_negative_determinants_avg / n_voxels
            ),
            f"{prefix}det_std_avg_along_other_dims": det_std_avg,
            f"{prefix}n_negative_determinants_crop_last_along_other_dims": (
                n_negative_determinants_crop_last
            ),
            f"{prefix}proportion_negative_determinants_crop_last_along_other_dims": (
                n_negative_determinants_crop_last / n_voxels
            ),
            f"{prefix}det_std_crop_last_along_other_dims": det_std_crop_last,
            f"{prefix}n_negative_determinants_central": n_negative_determinants_central,
            f"{prefix}proportion_negative_determinants_central": n_negative_determinants_central
            / n_voxels,
            f"{prefix}det_std_central": det_std_central,
        }

    @staticmethod
    def _determinant_metrics(
        displacement_field: Tensor, other_dims: str, central: bool
    ) -> tuple[int, float]:
        mapping = displacement_field[None] + generate_voxel_coordinate_grid(
            displacement_field.shape[1:],
            device=displacement_field.device,
            dtype=displacement_field.dtype,
        )
        jacobian_matrices = estimate_spatial_jacobian_matrices(
            volume=mapping, other_dims=other_dims, central=central
        )
        determinants = calculate_determinant(jacobian_matrices)
        n_neg_det = int((determinants < 0).sum())
        det_std = float(determinants.std())
        return n_neg_det, det_std

    @property
    def evaluation_inference_outputs(self) -> set[str]:
        return {
            "forward_displacement_field",
            "inverse_displacement_field",
        } | super().evaluation_inference_outputs
