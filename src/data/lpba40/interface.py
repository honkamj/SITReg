"""Interface for OASIS dataset"""

from typing import Any, Mapping

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from algorithm.affine_sampling import AffineTransformationSamplingArguments
from data.dataset import (
    VolumetricRegistrationInferenceDataset,
    VolumetricRegistrationTrainingDataset,
)
from data.interface import (
    InferenceDataArgs,
    TrainingDataLoader,
    TrainingDataLoaderArgs,
    VolumetricDataArgs,
)
from data.lpba40.data import LPBA40Data
from data.lpba40.inference import LPBA40InferenceFactory


def create_training_data_loader(
    data_config: Mapping[str, Any], args: TrainingDataLoaderArgs
) -> TrainingDataLoader:
    """Create oasis data loader for training"""
    data = LPBA40Data(
        data_root=args.data_root,
        both_directions=data_config["iterate_inference_pairs_in_both_directions"],
        file_type=data_config["file_type"],
        segmentation_file_type=data_config["segmentation_file_type"],
        included_segmentation_class_indices=data_config["included_segmentation_class_indices"],
        training_segmentation_class_index_groups=data_config.get(
            "training_segmentation_class_index_groups"
        ),
    )
    training_dataset = VolumetricRegistrationTrainingDataset(
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
    generate_new_variant = training_dataset.generate_new_variant
    if args.n_training_processes > 1:
        sampler: DistributedSampler | None = DistributedSampler(
            dataset=training_dataset,
            num_replicas=args.n_training_processes,
            rank=args.training_process_rank,
            shuffle=False,
        )
    else:
        sampler = None
    if data_config["batch_size"] % args.n_training_processes != 0:
        raise ValueError("Batch size must be divisible by the number of training processes")
    return TrainingDataLoader(
        data_loader=DataLoader(
            training_dataset,
            batch_size=data_config["batch_size"] // args.n_training_processes,
            num_workers=args.num_workers,
            drop_last=False,
            persistent_workers=True,
            sampler=sampler,
        ),
        generate_new_variant=generate_new_variant,
    )


def create_inference_data_factory(
    data_config: Mapping[str, Any], args: InferenceDataArgs
) -> LPBA40InferenceFactory:
    """Create oasis inference data loader factory"""
    data = LPBA40Data(
        data_root=args.data_root,
        both_directions=data_config["iterate_inference_pairs_in_both_directions"],
        file_type=data_config["file_type"],
        segmentation_file_type=data_config["segmentation_file_type"],
        included_segmentation_class_indices=data_config["included_segmentation_class_indices"],
    )
    return LPBA40InferenceFactory(
        dataset=VolumetricRegistrationInferenceDataset(
            data=data,
            data_args=VolumetricDataArgs.from_config(data_config),
            division=args.division,
        ),
        data_config=data_config,
    )
