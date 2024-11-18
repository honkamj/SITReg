"""SITReg registration training implementation"""

from itertools import chain, islice
from logging import getLogger
from typing import Any, Mapping, Optional, Sequence, TypeVar, cast

from composable_mapping import (
    ComposableMapping,
    CoordinateSystem,
    EnumeratedSamplingParameterCache,
    GridComposableMapping,
    ISampler,
    LinearInterpolator,
    mappable,
    samplable_volume,
)
from torch import Tensor, ones_like
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer

from application.base import BaseTrainingDefinition
from application.interface import TrainingDefinitionArgs
from loss.interface import (
    create_losses,
    create_regularity_loss,
    create_segmentation_loss,
    create_similarity_loss,
    get_combined_loss,
)
from loss.similarity import MeanSquaredError
from model.sitreg import SITReg
from model.sitreg.model import MappingPair
from util.checked_type_casting import to_two_tuple

logger = getLogger(__name__)
T = TypeVar("T")


class SITRegTraining(BaseTrainingDefinition):
    """Training for SITReg"""

    def __init__(
        self,
        model: SITReg,
        application_config: Mapping[str, Any],
        args: TrainingDefinitionArgs,
    ) -> None:
        super().__init__(application_config)
        self._devices = args.devices
        training_config = application_config["training"]
        optimizer_config = training_config["optimizer"]
        if args.n_training_processes > 1:
            distributed_training_model = DistributedDataParallel(model)
            self._training_model: Module = distributed_training_model
            self._model: SITReg = cast(SITReg, distributed_training_model.module)
        else:
            self._training_model = model
            self._model = model
        self._optimizer = Adam(
            params=list(self._training_model.parameters()),
            lr=optimizer_config["learning_rate"],
            betas=to_two_tuple(float, optimizer_config["betas"]),
        )
        self._n_epochs = training_config["n_epochs"]
        loss_config = training_config["loss"]
        self._multiscale_similarity_losses = {
            level_index: create_losses(similarity_losses_for_scale, create_similarity_loss)
            for level_index, similarity_losses_for_scale in enumerate(loss_config["similarity"])
            if similarity_losses_for_scale is not None and similarity_losses_for_scale
        }
        self._multiscale_regularity_losses = {
            level_index: create_losses(regularity_losses_for_scale, create_regularity_loss)
            for level_index, regularity_losses_for_scale in enumerate(loss_config["regularity"])
            if regularity_losses_for_scale is not None and regularity_losses_for_scale
        }
        self._similarity_loss_levels = list(set(self._multiscale_similarity_losses.keys()))
        self._regularity_loss_levels = list(set(self._multiscale_regularity_losses.keys()))
        self._regularize_with_affine = loss_config.get("regularize_with_affine", True)
        self._ignore_mask = application_config["training"].get("ignore_mask", False)
        self._sampling_parameter_cache = EnumeratedSamplingParameterCache()

    def _handle_mask(self, mask: Tensor) -> Tensor:
        if self._ignore_mask:
            return ones_like(mask, device=self._devices[0])
        return mask

    def update_weights(self, batch: Any) -> Mapping[str, float]:
        with self._sampling_parameter_cache:
            loss, loss_dict = self._compute_losses(batch)
        self._optimizer.zero_grad(set_to_none=True)
        logger.debug("Starting backward pass")
        loss.backward()
        self._optimizer.step()
        return loss_dict

    @staticmethod
    def _split_sequence_into_chunks(
        sequence: Sequence[T], chunk_sizes: Sequence[int]
    ) -> Sequence[Sequence[T]]:
        sequence_iterator = iter(sequence)
        return [list(islice(sequence_iterator, chunk_size)) for chunk_size in chunk_sizes]

    def _obtain_deformations(
        self,
        image_1: Tensor,
        image_2: Tensor,
        mappings_for_levels_lists: Sequence[Sequence[tuple[int, bool]]],
    ) -> Sequence[Sequence[MappingPair]]:
        deformations: list[MappingPair]
        deformations = self._training_model(
            image_1,
            image_2,
            mappings_for_levels=list(chain.from_iterable(mappings_for_levels_lists)),
        )
        return self._split_sequence_into_chunks(
            deformations,
            [len(mappings_for_levels) for mappings_for_levels in mappings_for_levels_lists],
        )

    def _compute_losses(
        self,
        batch: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        (image_1, mask_1, augmentation_1), (image_2, mask_2, augmentation_2) = batch

        image_1 = image_1.to(self._devices[0])
        mask_1 = self._handle_mask(mask_1.to(self._devices[0]))
        image_2 = image_2.to(self._devices[0])
        mask_2 = self._handle_mask(mask_2.to(self._devices[0]))

        augmentation_1_mapping: ComposableMapping = augmentation_1.as_mapping().cast(
            device=self._devices[0]
        )
        augmentation_2_mapping: ComposableMapping = augmentation_2.as_mapping().cast(
            device=self._devices[0]
        )

        image_1_mapping = self._image_as_mapping(image_1, mask_1) @ augmentation_1_mapping
        image_2_mapping = self._image_as_mapping(image_2, mask_2) @ augmentation_2_mapping

        regularity_deformations, similarity_deformations = self._obtain_deformations(
            image_1=image_1_mapping.sample().generate_values(),
            image_2=image_2_mapping.sample().generate_values(),
            mappings_for_levels_lists=(
                [
                    (level_index, self._regularize_with_affine)
                    for level_index in self._regularity_loss_levels
                ],
                [(level_index, True) for level_index in self._similarity_loss_levels],
            ),
        )

        loss, loss_dict = self._compute_regularity_losses(
            regularity_deformations=regularity_deformations,
        )
        loss, loss_dict = self._compute_similarity_losses(
            similarity_deformations=similarity_deformations,
            image_1_mapping=image_1_mapping,
            image_2_mapping=image_2_mapping,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
        )
        assert loss is not None
        assert loss_dict is not None

        return loss, loss_dict

    def _compute_regularity_losses(
        self,
        regularity_deformations: Sequence[MappingPair],
        previous_loss: Tensor | None = None,
        previous_loss_dict=None,
        tensor_loss_dict: bool = False,
    ) -> tuple[Tensor | None, Any]:
        loss = previous_loss
        loss_dict = previous_loss_dict
        for level_index, mapping_pair in zip(self._regularity_loss_levels, regularity_deformations):
            regularity_losses = self._multiscale_regularity_losses[level_index]
            loss, loss_dict = get_combined_loss(
                regularity_losses,
                prefix="forward_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
                tensor_loss_dict=tensor_loss_dict,
            )(
                mapping=mapping_pair.forward_mapping,
            )
            loss, loss_dict = get_combined_loss(
                regularity_losses,
                prefix="inverse_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
                tensor_loss_dict=tensor_loss_dict,
            )(
                mapping=mapping_pair.inverse_mapping,
            )
        return loss, loss_dict

    def _compute_similarity_losses(
        self,
        similarity_deformations: Sequence[MappingPair],
        image_1_mapping: GridComposableMapping,
        image_2_mapping: GridComposableMapping,
        loss_kwargs: Mapping[str, Mapping[str, Mapping[str, Tensor]]] | None = None,
        previous_loss: Tensor | None = None,
        previous_loss_dict=None,
        tensor_loss_dict: bool = False,
    ) -> tuple[Tensor | None, Any]:
        loss = previous_loss
        loss_dict = previous_loss_dict
        for level_index, mapping_pair in zip(self._similarity_loss_levels, similarity_deformations):
            deformed_image_1 = image_1_mapping @ mapping_pair.forward_mapping
            deformed_image_2 = image_2_mapping @ mapping_pair.inverse_mapping
            similarity_losses = self._multiscale_similarity_losses[level_index]
            loss, loss_dict = get_combined_loss(
                similarity_losses,
                prefix="forward_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
                default_kwargs_per_name=loss_kwargs,
                tensor_loss_dict=tensor_loss_dict,
            )(
                image_1=deformed_image_1.sample(),
                image_2=image_2_mapping.sample(),
            )
            loss, loss_dict = get_combined_loss(
                similarity_losses,
                prefix="inverse_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
                default_kwargs_per_name=loss_kwargs,
                tensor_loss_dict=tensor_loss_dict,
            )(
                image_1=deformed_image_2.sample(),
                image_2=image_1_mapping.sample(),
            )
        return loss, loss_dict

    def get_optimizers(self) -> Mapping[str, Optimizer]:
        return {"optimizer": self._optimizer}

    def get_modules(self) -> Mapping[str, Module]:
        return {"registration_network": self._model}

    def _image_as_mapping(
        self, image: Tensor, mask: Optional[Tensor] = None, sampler: ISampler = LinearInterpolator()
    ) -> GridComposableMapping:
        return samplable_volume(
            image,
            mask=mask,
            coordinate_system=self._model.image_coordinate_system,
            sampler=sampler,
        )


class SITRegLandmarkTraining(SITRegTraining):
    """Training for SITReg with additional landmark supervision"""

    def __init__(
        self, model: SITReg, application_config: Mapping[str, Any], args: TrainingDefinitionArgs
    ) -> None:
        super().__init__(model, application_config, args)
        self._landmark_coordinate_system = CoordinateSystem.voxel(
            spatial_shape=application_config["model"]["input_shape"],
            device=self._devices[0],
        )
        self._landmark_loss_weight = application_config["training"]["loss"]["landmark_weight"]
        self._landmark_loss = MeanSquaredError()

    def _compute_losses(
        self,
        batch: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        (image_1, mask_1, augmentation_1, landmarks_1), (
            image_2,
            mask_2,
            augmentation_2,
            landmarks_2,
        ) = batch

        image_1 = image_1.to(self._devices[0])
        mask_1 = self._handle_mask(mask_1.to(self._devices[0]))
        landmarks_1 = landmarks_1.to(self._devices[0])
        image_2 = image_2.to(self._devices[0])
        mask_2 = self._handle_mask(mask_2.to(self._devices[0]))
        landmarks_2 = landmarks_2.to(self._devices[0])

        augmentation_1_mapping: ComposableMapping = augmentation_1.as_mapping().cast(
            device=self._devices[0]
        )
        augmentation_2_mapping: ComposableMapping = augmentation_2.as_mapping().cast(
            device=self._devices[0]
        )

        image_1_mapping = self._image_as_mapping(image_1, mask_1) @ augmentation_1_mapping
        image_2_mapping = self._image_as_mapping(image_2, mask_2) @ augmentation_2_mapping

        regularity_deformations, similarity_deformations, landmark_deformations = (
            self._obtain_deformations(
                image_1=image_1_mapping.sample().generate_values(),
                image_2=image_2_mapping.sample().generate_values(),
                mappings_for_levels_lists=(
                    [
                        (level_index, self._regularize_with_affine)
                        for level_index in self._regularity_loss_levels
                    ],
                    [(level_index, True) for level_index in self._similarity_loss_levels],
                    [(0, True)],
                ),
            )
        )
        landmark_deformation = landmark_deformations[0]

        mappable_landmarks_1 = mappable(landmarks_1)
        mappable_landmarks_2 = mappable(landmarks_2)
        transformed_landmarks_1 = (
            self._landmark_coordinate_system.from_voxel_coordinates
            @ self._model.transformation_coordinate_system.to_voxel_coordinates
            @ landmark_deformation.inverse_mapping
            @ self._model.transformation_coordinate_system.from_voxel_coordinates
            @ self._landmark_coordinate_system.to_voxel_coordinates
        )(mappable_landmarks_1)
        transformed_landmarks_2 = (
            self._landmark_coordinate_system.from_voxel_coordinates
            @ self._model.transformation_coordinate_system.to_voxel_coordinates
            @ landmark_deformation.forward_mapping
            @ self._model.transformation_coordinate_system.from_voxel_coordinates
            @ self._landmark_coordinate_system.to_voxel_coordinates
        )(mappable_landmarks_2)

        landmark_loss = (
            self._landmark_loss(transformed_landmarks_1, mappable_landmarks_2)
            + self._landmark_loss(transformed_landmarks_2, mappable_landmarks_1)
        ) / 2
        loss = self._landmark_loss_weight * landmark_loss
        loss_dict = {
            "loss": loss.item(),
            "landmark_loss": landmark_loss.item(),
        }

        loss, loss_dict = self._compute_regularity_losses(
            regularity_deformations=regularity_deformations,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
        )
        loss, loss_dict = self._compute_similarity_losses(
            similarity_deformations=similarity_deformations,
            image_1_mapping=image_1_mapping,
            image_2_mapping=image_2_mapping,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
        )
        assert loss is not None
        assert loss_dict is not None

        return loss, loss_dict


class SITRegSegmentationTraining(SITRegTraining):
    """Training for SITReg with additional segmentation supervision"""

    def __init__(
        self,
        model: SITReg,
        application_config: Mapping[str, Any],
        args: TrainingDefinitionArgs,
    ) -> None:
        super().__init__(
            model=model,
            application_config=application_config,
            args=args,
        )
        training_config = application_config["training"]
        loss_config = training_config["loss"]
        self._multiscale_segmentation_losses = {
            level_index: create_losses(segmentation_losses_for_scale, create_segmentation_loss)
            for level_index, segmentation_losses_for_scale in enumerate(loss_config["segmentation"])
            if segmentation_losses_for_scale is not None and segmentation_losses_for_scale
        }
        self._segmentation_loss_levels = list(set(self._multiscale_segmentation_losses.keys()))

    def _compute_losses(
        self,
        batch: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        (image_1, mask_1, augmentation_1, seg_1), (image_2, mask_2, augmentation_2, seg_2) = batch

        image_1 = image_1.to(self._devices[0])
        mask_1 = self._handle_mask(mask_1.to(self._devices[0]))
        seg_1 = seg_1.to(self._devices[0])
        image_2 = image_2.to(self._devices[0])
        mask_2 = self._handle_mask(mask_2.to(self._devices[0]))
        seg_2 = seg_2.to(self._devices[0])

        augmentation_1_mapping: ComposableMapping = augmentation_1.as_mapping().cast(
            device=self._devices[0]
        )
        augmentation_2_mapping: ComposableMapping = augmentation_2.as_mapping().cast(
            device=self._devices[0]
        )
        image_1_mapping = self._image_as_mapping(image_1, mask_1) @ augmentation_1_mapping
        image_2_mapping = self._image_as_mapping(image_2, mask_2) @ augmentation_2_mapping
        seg_1_mapping = self._image_as_mapping(seg_1, mask_1) @ augmentation_1_mapping
        seg_2_mapping = self._image_as_mapping(seg_2, mask_2) @ augmentation_2_mapping

        regularity_deformations, similarity_deformations, segmentation_deformations = (
            self._obtain_deformations(
                image_1_mapping.sample().generate_values(),
                image_2_mapping.sample().generate_values(),
                mappings_for_levels_lists=(
                    [
                        (level_index, self._regularize_with_affine)
                        for level_index in self._regularity_loss_levels
                    ],
                    [(level_index, True) for level_index in self._similarity_loss_levels],
                    [(level_index, True) for level_index in self._segmentation_loss_levels],
                ),
            )
        )

        loss, loss_dict = self._compute_regularity_losses(
            regularity_deformations=regularity_deformations,
        )
        loss, loss_dict = self._compute_similarity_losses(
            similarity_deformations=similarity_deformations,
            image_1_mapping=image_1_mapping,
            image_2_mapping=image_2_mapping,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
        )
        loss, loss_dict = self._compute_segmentation_losses(
            segmentation_deformations=segmentation_deformations,
            seg_1_mapping=seg_1_mapping,
            seg_2_mapping=seg_2_mapping,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
        )
        assert loss is not None
        assert loss_dict is not None

        return loss, loss_dict

    def _compute_segmentation_losses(
        self,
        segmentation_deformations: Sequence[MappingPair],
        seg_1_mapping: GridComposableMapping,
        seg_2_mapping: GridComposableMapping,
        previous_loss: Tensor | None = None,
        previous_loss_dict: dict[str, float] | None = None,
    ) -> tuple[Tensor | None, dict[str, float] | None]:
        loss = previous_loss
        loss_dict = previous_loss_dict
        for level_index, mapping_pair in zip(
            self._similarity_loss_levels, segmentation_deformations
        ):
            resampled_seg_1_mapping = seg_1_mapping @ mapping_pair.forward_mapping
            resampled_seg_2_mapping = seg_2_mapping @ mapping_pair.inverse_mapping
            segmentation_losses = self._multiscale_segmentation_losses[level_index]
            loss, loss_dict = get_combined_loss(
                segmentation_losses,
                prefix="forward_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                seg_1=resampled_seg_1_mapping.sample(),
                seg_2=seg_2_mapping.sample(),
            )
            loss, loss_dict = get_combined_loss(
                segmentation_losses,
                prefix="inverse_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                seg_1=resampled_seg_2_mapping.sample(),
                seg_2=seg_1_mapping.sample(),
            )
        return loss, loss_dict
