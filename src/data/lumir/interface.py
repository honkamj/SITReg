"""Interface for OASIS dataset"""

from typing import Any, Mapping

from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from algorithm.affine_sampling import AffineTransformationSamplingArguments
from data.dataset import (
    VolumetricRegistrationInferenceDataset,
    VolumetricRegistrationSegmentationTrainingDataset,
    VolumetricRegistrationTrainingDataset,
    VolumetricRegistrationTrainingDatasetWithReplacement,
)
from data.interface import (
    InferenceDataArgs,
    IVariantDataset,
    TrainingDataLoader,
    TrainingDataLoaderArgs,
    VolumetricDataArgs,
)

from .data import LumirData
from .inference import LumirInferenceFactory


def _create_training_dataset(data: LumirData, data_config: Mapping[str, Any]) -> IVariantDataset:
    include_segmentations_for_training = data_config.get(
        "include_segmentations_for_training", False
    )
    n_items_per_step = data_config.get("n_training_items_per_step", 2)
    if n_items_per_step != 2:
        if include_segmentations_for_training:
            raise ValueError(
                "n_training_items_per_step must be 2 when including segmentations for training"
            )
        return VolumetricRegistrationTrainingDatasetWithReplacement(
            data=data,
            data_args=VolumetricDataArgs.from_config(data_config),
            n_steps_per_epoch=data_config["n_training_items_per_epoch"],
            n_training_cases=data_config["n_training_cases"],
            seed=data_config["seed"],
            n_items_per_step=n_items_per_step,
        )
    if not include_segmentations_for_training:
        dataset_class = VolumetricRegistrationTrainingDataset
    else:
        dataset_class = VolumetricRegistrationSegmentationTrainingDataset
    return dataset_class(
        data=data,
        data_args=VolumetricDataArgs.from_config(data_config),
        seed=data_config["seed"],
        pairs_per_epoch=data_config["training_pairs_per_epoch"],
        n_training_cases=data_config["n_training_cases"],
        affine_augmentation_arguments=(
            AffineTransformationSamplingArguments(**data_config["affine_augmentation_arguments"])
            if data_config.get("affine_augmentation_arguments") is not None
            else None
        ),
        affine_augmentation_prob=data_config.get("affine_augmentation_prob"),
    )


def create_training_data_loader(
    data_config: Mapping[str, Any], args: TrainingDataLoaderArgs
) -> TrainingDataLoader:
    """Create oasis data loader for training"""
    data = LumirData(
        data_root=args.data_root,
        both_directions=data_config["iterate_inference_pairs_in_both_directions"],
        included_segmentation_class_indices=data_config["included_segmentation_class_indices"],
        training_segmentation_class_index_groups=None,
        use_body_mask_as_mask_with_erosion=data_config.get("use_body_mask_as_mask_with_erosion"),
    )
    training_dataset = _create_training_dataset(data=data, data_config=data_config)
    batch_size = data_config["batch_size"]
    training_process_chunk_size = data_config.get("training_process_chunk_size", 1)
    n_training_process_chunks = args.n_training_processes // training_process_chunk_size
    if n_training_process_chunks > 1:
        chunk_rank = args.training_process_rank // training_process_chunk_size
        sampler: Sampler | None = DistributedSampler(
            dataset=training_dataset,
            num_replicas=n_training_process_chunks,
            rank=chunk_rank,
            shuffle=False,
        )
        if batch_size % n_training_process_chunks != 0:
            raise ValueError("Batch size must be divisible by the number of training processes")
        batch_size = batch_size // n_training_process_chunks
    else:
        sampler = None
    generate_new_variant = training_dataset.generate_new_variant
    return TrainingDataLoader(
        data_loader=DataLoader(
            training_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            persistent_workers=True,
            sampler=sampler,
        ),
        generate_new_variant=generate_new_variant,
    )


def create_inference_data_factory(
    data_config: Mapping[str, Any], args: InferenceDataArgs
) -> LumirInferenceFactory:
    """Create lumir inference data loader factory"""
    data = LumirData(
        data_root=args.data_root,
        both_directions=data_config["iterate_inference_pairs_in_both_directions"],
        included_segmentation_class_indices=data_config["included_segmentation_class_indices"],
    )
    return LumirInferenceFactory(
        dataset=VolumetricRegistrationInferenceDataset(
            data=data,
            data_args=VolumetricDataArgs.from_config(data_config),
            division=args.division,
        ),
        data_config=data_config,
    )
