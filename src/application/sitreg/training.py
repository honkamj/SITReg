"""SITReg registration training implementation"""

from logging import getLogger
from typing import Any, Mapping, cast

from torch import Tensor
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer

from algorithm.composable_mapping.factory import ComposableFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.composable_mapping.interface import IComposableMapping
from algorithm.interpolator import LinearInterpolator
from application.base import BaseTrainingDefinition
from application.interface import TrainingDefinitionArgs
from loss.interface import (
    create_losses,
    create_regularity_loss,
    create_similarity_loss,
    get_combined_loss,
)
from model.sitreg import SITReg
from model.sitreg.model import MappingPair
from util.checked_type_casting import to_two_tuple
from util.ndimensional_operators import avg_pool_nd_function

logger = getLogger(__name__)


class SITRegTraining(BaseTrainingDefinition):
    """Training for SITReg"""

    def __init__(
        self,
        model: SITReg,
        application_config: Mapping[str, Any],
        args: TrainingDefinitionArgs,
    ) -> None:
        super().__init__(application_config)
        self._device = args.device
        training_config = application_config["training"]
        optimizer_config = training_config["optimizer"]
        if args.n_training_processes > 1:
            distributed_training_model = DistributedDataParallel(model)
            self._training_model: Module = distributed_training_model
            self._model: SITReg = cast(SITReg, distributed_training_model.module)
            self._zero_redundacy_optimizer: ZeroRedundancyOptimizer | None = (
                ZeroRedundancyOptimizer(
                    params=self._training_model.parameters(),
                    optimizer_class=Adam,
                    lr=optimizer_config["learning_rate"],
                    betas=to_two_tuple(float, optimizer_config["betas"]),
                )
            )
            self._optimizer: Optimizer = self._zero_redundacy_optimizer
        else:
            self._training_model = model
            self._optimizer = Adam(
                params=self._training_model.parameters(),
                lr=optimizer_config["learning_rate"],
                betas=to_two_tuple(float, optimizer_config["betas"]),
            )
            self._zero_redundacy_optimizer = None
            self._model = model
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
        self._average_pool = avg_pool_nd_function(len(application_config["model"]["input_shape"]))

    def update_weights(self, batch: tuple[Tensor, Tensor]) -> Mapping[str, float]:
        (image_1, image_1_mask), (image_2, image_2_mask) = batch
        image_1 = image_1.to(self._device)
        image_1_mask = image_1_mask.to(self._device)
        image_2 = image_2.to(self._device)
        image_2_mask = image_2_mask.to(self._device)
        dtype = image_1.dtype
        deformations: list[MappingPair]
        deformations = self._training_model(
            image_1,
            image_2,
            mappings_for_levels=(
                [
                    (level_index, self._regularize_with_affine)
                    for level_index in self._regularity_loss_levels
                ]
                + [(level_index, True) for level_index in self._similarity_loss_levels]
            ),
        )
        regularity_deformations = deformations[: len(self._regularity_loss_levels)]
        similarity_deformations = deformations[-len(self._similarity_loss_levels) :]
        assert len(regularity_deformations) + len(similarity_deformations) == len(deformations)
        image_1_mapping = self._image_as_mapping(image_1, image_1_mask)
        image_2_mapping = self._image_as_mapping(image_2, image_2_mask)

        loss: Tensor | None = None
        loss_dict: dict[str, float] | None = None

        for level_index, mapping_pair in zip(self._regularity_loss_levels, regularity_deformations):
            regularity_losses = self._multiscale_regularity_losses[level_index]
            loss, loss_dict = get_combined_loss(
                regularity_losses,
                prefix="forward_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                mapping=mapping_pair.forward_mapping,
                coordinate_system=self._model.image_coordinate_system,
                device=self._device,
                dtype=dtype,
            )
            loss, loss_dict = get_combined_loss(
                regularity_losses,
                prefix="inverse_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                mapping=mapping_pair.inverse_mapping,
                coordinate_system=self._model.image_coordinate_system,
                device=self._device,
                dtype=dtype,
            )
        for level_index, mapping_pair in zip(self._similarity_loss_levels, similarity_deformations):
            resampled_image_1_mapping = image_1_mapping.compose(mapping_pair.forward_mapping)
            resampled_image_2_mapping = image_2_mapping.compose(mapping_pair.inverse_mapping)
            similarity_losses = self._multiscale_similarity_losses[level_index]
            loss, loss_dict = get_combined_loss(
                similarity_losses,
                prefix="forward_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                image_1=resampled_image_1_mapping(self._model.image_coordinate_system.grid),
                image_2=image_2_mapping(self._model.image_coordinate_system.grid),
                device=self._device,
                dtype=dtype,
            )
            loss, loss_dict = get_combined_loss(
                similarity_losses,
                prefix="inverse_",
                previous_loss=loss,
                previous_loss_dict=loss_dict,
                weight=1 / 2,
            )(
                image_1=resampled_image_2_mapping(self._model.image_coordinate_system.grid),
                image_2=image_1_mapping(self._model.image_coordinate_system.grid),
                device=self._device,
                dtype=dtype,
            )
        assert loss is not None
        assert loss_dict is not None
        self._optimizer.zero_grad(set_to_none=True)
        logger.debug("Starting backward pass")
        loss.backward()
        self._optimizer.step()
        return loss_dict

    def get_optimizers(self) -> Mapping[str, Optimizer]:
        return {"optimizer": self._optimizer}

    def get_modules(self) -> Mapping[str, Module]:
        return {"registration_network": self._model}

    def _image_as_mapping(self, image: Tensor, mask: Tensor) -> IComposableMapping:
        return ComposableFactory.create_volume(
            data=image,
            mask=mask,
            coordinate_system=self._model.image_coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=LinearInterpolator(), mask_outside_fov=True
            ),
        )

    def before_save(self, saving_process_rank: int) -> None:
        if self._zero_redundacy_optimizer is not None:
            self._zero_redundacy_optimizer.consolidate_state_dict(to=saving_process_rank)
