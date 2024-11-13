"""SITReg registration training implementation"""

from itertools import chain, islice
from logging import getLogger
from typing import Any, Mapping, Sequence, TypeVar, cast

from composable_mapping import (
    DataFormat,
    EnumeratedSamplingParameterCache,
    GridComposableMapping,
    LinearInterpolator,
    samplable_volume,
)
from torch import Tensor, cat, no_grad, ones_like
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from application.base import BaseTrainingDefinition
from application.interface import TrainingDefinitionArgs
from loss.interface import (
    create_losses,
    create_regularity_loss,
    create_segmentation_loss,
    create_similarity_loss,
    get_combined_loss,
)
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
        (image_1, mask_1), (image_2, mask_2) = batch

        image_1 = image_1.to(self._devices[0])
        mask_1 = self._handle_mask(mask_1.to(self._devices[0]))
        image_2 = image_2.to(self._devices[0])
        mask_2 = self._handle_mask(mask_2.to(self._devices[0]))

        image_1_mapping = self._image_as_mapping(image_1, mask_1)
        image_2_mapping = self._image_as_mapping(image_2, mask_2)

        regularity_deformations, similarity_deformations = self._obtain_deformations(
            image_1=image_1,
            image_2=image_2,
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

    def _image_as_mapping(self, image: Tensor, mask: Tensor) -> GridComposableMapping:
        return samplable_volume(
            image,
            mask=mask,
            coordinate_system=self._model.image_coordinate_system,
            sampler=LinearInterpolator(),
        )


class SITRegSemiSupervisedTraining(SITRegTraining):
    """Training for SITReg with segmentation maps"""

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
        (image_1, image_1_mask, image_1_seg), (
            image_2,
            image_2_mask,
            image_2_seg,
        ) = batch

        image_1 = image_1.to(self._devices[0])
        image_1_mask = image_1_mask.to(self._devices[0])
        image_1_seg = image_1_seg.to(self._devices[0])
        image_2 = image_2.to(self._devices[0])
        image_2_mask = image_2_mask.to(self._devices[0])
        image_2_seg = image_2_seg.to(self._devices[0])

        regularity_deformations, similarity_deformations, segmentation_deformations = (
            self._obtain_deformations(
                image_1,
                image_2,
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
        image_1_mapping = self._image_as_mapping(image_1, image_1_mask)
        image_2_mapping = self._image_as_mapping(image_2, image_2_mask)

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
            seg_1_mapping=self._image_as_mapping(image_1_seg, image_1_mask),
            seg_2_mapping=self._image_as_mapping(image_2_seg, image_2_mask),
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


class SITRegRegularityFineTuningTraining(BaseTrainingDefinition):
    """Regularity fine tuning training for SITReg"""

    def __init__(
        self,
        model: SITReg,
        ndv_model: Module,
        application_config: Mapping[str, Any],
        args: TrainingDefinitionArgs,
    ) -> None:
        super().__init__(application_config)
        self._devices = args.devices
        training_config = application_config["training"]
        optimizer_config = training_config["optimizer"]
        self._model: SITReg = model
        if args.n_training_processes > 1:
            distributed_training_model = DistributedDataParallel(ndv_model)
            self._training_ndv_model: Module = distributed_training_model
            self._ndv_model = distributed_training_model.module
        else:
            self._training_ndv_model = ndv_model
            self._ndv_model = ndv_model
        self._optimizer = Adam(
            params=list(self._ndv_model.parameters()),
            lr=optimizer_config["learning_rate"],
            betas=to_two_tuple(float, optimizer_config["betas"]),
        )
        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer,
            gamma=optimizer_config["lr_decay"],
        )
        self._n_epochs = training_config["n_epochs"]
        loss_config = training_config["loss"]
        self._regularity_losses = create_losses(loss_config["regularity"], create_regularity_loss)
        self._process_rank = args.training_process_rank

    def update_weights(self, batch: Any) -> Mapping[str, float]:
        (image_1, image_1_mask), (image_2, image_2_mask) = batch

        if image_1.size(0) > 1:
            raise RuntimeError("NDV training only supports batch size 1")

        image_1 = image_1.to(self._devices[0])
        image_1_mask = image_1_mask.to(self._devices[0])
        image_2 = image_2.to(self._devices[0])
        image_2_mask = image_2_mask.to(self._devices[0])

        with no_grad():
            deformations: MappingPair
            deformations = self._model(
                image_1,
                image_2,
                mappings_for_levels=((0, True),),
            )[0]
            forward_ddf = deformations.forward_mapping.sample(
                DataFormat.voxel_displacements()
            ).generate_values()
            inverse_ddf = deformations.inverse_mapping.sample(
                DataFormat.voxel_displacements()
            ).generate_values()
            ddfs = cat([forward_ddf, inverse_ddf], dim=0)

        modifications = self._training_ndv_model(ddfs)
        forward_modification, inverse_modification = modifications.chunk(2, dim=0)
        modified_forward_ddf = forward_ddf + forward_modification
        modified_inverse_ddf = inverse_ddf + inverse_modification

        identity_loss = (
            (forward_modification.square().sum(dim=1) + 1e-6).sqrt().mean()
            + (inverse_modification.square().sum(dim=1) + 1e-6).sqrt().mean()
        ) / 2

        modified_forward_deformation = self._obtain_ddf_as_mapping(modified_forward_ddf)
        modified_inverse_deformation = self._obtain_ddf_as_mapping(modified_inverse_ddf)

        loss, loss_dict = get_combined_loss(
            self._regularity_losses,
            prefix="forward_",
            weight=1 / 2,
        )(
            mapping=modified_forward_deformation,
        )
        loss, loss_dict = get_combined_loss(
            self._regularity_losses,
            prefix="inverse_",
            previous_loss=loss,
            previous_loss_dict=loss_dict,
            weight=1 / 2,
        )(
            mapping=modified_inverse_deformation,
        )

        loss = identity_loss + loss

        loss_dict["identity"] = identity_loss.item()
        loss_dict["loss"] = loss.item()

        self._optimizer.zero_grad(set_to_none=True)
        logger.debug("Starting backward pass")
        loss.backward()
        self._optimizer.step()
        self._lr_scheduler.step()
        return loss_dict

    def _obtain_ddf_as_mapping(self, ddf: Tensor) -> GridComposableMapping:
        mapping = samplable_volume(
            ddf,
            coordinate_system=self._model.transformation_coordinate_system,
            sampler=LinearInterpolator(),
            data_format=DataFormat.voxel_displacements(),
        )
        return mapping

    def get_optimizers(self) -> Mapping[str, Optimizer | LRScheduler]:
        return {"optimizer": self._optimizer, "lr_scheduler": self._lr_scheduler}

    def get_modules(self) -> Mapping[str, Module]:
        return {"registration_network": self._model, "ndv_network": self._ndv_model}
