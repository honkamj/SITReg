"""SITReg registration training implementation"""

from logging import getLogger
from typing import Any, Mapping, Sequence, TypeVar

from composable_mapping import (
    ComposableMapping,
    DataFormat,
    GridComposableMapping,
    Identity,
    LinearInterpolator,
    samplable_volume,
)
from numpy import interp
from torch import Tensor, empty_like, inverse, ones
from torch.distributed import (
    broadcast,
    get_group_rank,
    get_process_group_ranks,
    new_group,
)

from algorithm.multiply_gradient import multiply_backward
from application.interface import TrainingDefinitionArgs
from loss.interface import average_tensor_loss_dict
from model.sitreg import SITReg
from model.sitreg.model import MappingPair

from .training import SITRegTraining

logger = getLogger(__name__)


T = TypeVar("T")


class SITRegDistributedConsistencyTraining(SITRegTraining):
    """Distributed training for SITReg with population consistency loss"""

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
        if len(self._devices) > 1:
            raise ValueError("Please specify only one GPU per process for distributed training.")
        self._consistency_loss_weight_per_epoch: Sequence[float] = application_config["training"][
            "loss"
        ]["consistency_loss_weight_per_epoch"]
        self._current_step = 0
        if args.n_training_processes % 3 != 0:
            raise ValueError(
                "Distributed training has been implemented for a multiple of 3 training processes"
            )
        n_groups = args.n_training_processes // 3
        group_index = args.training_process_rank // 3
        groups = [
            new_group(ranks=list(range(3 * index, 3 * index + 3))) for index in range(n_groups)
        ]
        self._training_group = groups[group_index]
        self._training_group_indices = get_process_group_ranks(self._training_group)
        self._process_within_group_rank = get_group_rank(
            self._training_group, args.training_process_rank
        )
        self._process_rank = args.training_process_rank
        # Solves https://github.com/pytorch/pytorch/issues/90613
        inverse(ones((1, 1), device=self._devices[0]))

    def update_weights(self, batch: Any) -> Mapping[str, float]:
        self._current_step += 1
        return super().update_weights(batch)

    def start_of_epoch(self, epoch: int, n_steps: int) -> None:
        super().start_of_epoch(epoch, n_steps)
        self._current_step = 0

    def _get_float_epoch(self) -> float:
        return self._current_epoch + self._current_step / self._n_steps_per_epoch

    def _get_loss_weight(self, loss_weight_per_epoch: Sequence[float]) -> float:
        return float(
            interp(
                self._get_float_epoch(),
                range(len(loss_weight_per_epoch)),
                loss_weight_per_epoch,
            )
        )

    def _get_consistency_loss_weight(self) -> float:
        return self._get_loss_weight(self._consistency_loss_weight_per_epoch)

    def _compute_pairwise_losses(
        self,
        image_1_mapping: GridComposableMapping,
        image_2_mapping: GridComposableMapping,
    ) -> tuple[Tensor, Mapping[str, Tensor], Tensor, Tensor]:
        regularity_deformations, similarity_deformations, consistency_deformations = (
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
        consistency_deformation = consistency_deformations[-1]
        loss, loss_dict = self._compute_regularity_losses(
            regularity_deformations=regularity_deformations,
            tensor_loss_dict=True,
        )
        loss, loss_dict = self._compute_similarity_losses(
            similarity_deformations=similarity_deformations,
            image_1_mapping=image_1_mapping,
            image_2_mapping=image_2_mapping,
            previous_loss=loss,
            previous_loss_dict=loss_dict,
            tensor_loss_dict=True,
        )
        consistency_forward_ddf = consistency_deformation.forward_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()
        consistency_inverse_ddf = consistency_deformation.inverse_mapping.sample(
            DataFormat.voxel_displacements()
        ).generate_values()
        assert loss is not None
        return loss, loss_dict, consistency_forward_ddf, consistency_inverse_ddf

    def _compute_losses(
        self,
        batch: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        (
            (image_1, mask_1, augmentation_1),
            (image_2, mask_2, augmentation_2),
            (image_3, mask_3, augmentation_3),
        ) = batch

        mask_1 = self._handle_mask(mask_1.to(self._devices[0]))
        mask_2 = self._handle_mask(mask_2.to(self._devices[0]))
        mask_3 = self._handle_mask(mask_3.to(self._devices[0]))

        images = [image_1, image_2, image_3]
        masks = [mask_1, mask_2, mask_3]
        augmentations: list[ComposableMapping] = [
            augmentation_1.as_mapping(),
            augmentation_2.as_mapping(),
            augmentation_3.as_mapping(),
        ]

        pair_augmentation_1_mapping = augmentations[self._process_within_group_rank % 3].cast(
            device=self._devices[0]
        )
        pair_augmentation_2_mapping = augmentations[(self._process_within_group_rank + 1) % 3].cast(
            device=self._devices[0]
        )

        loss, tensor_loss_dict, forward_ddf, inverse_ddf = self._compute_pairwise_losses(
            image_1_mapping=self._image_as_mapping(
                images[self._process_within_group_rank % 3].to(self._devices[0]),
                mask=masks[self._process_within_group_rank % 3],
            )
            @ pair_augmentation_1_mapping,
            image_2_mapping=self._image_as_mapping(
                images[(self._process_within_group_rank + 1) % 3].to(self._devices[0]),
                mask=masks[(self._process_within_group_rank + 1) % 3],
            )
            @ pair_augmentation_2_mapping,
        )
        loss = loss.mean()
        loss_dict = average_tensor_loss_dict(tensor_loss_dict)

        forward_ddf = (
            self._ddf_as_mapping(forward_ddf)
            .sample(data_format=DataFormat.voxel_displacements())
            .generate_values()
            .contiguous()
        )
        inverse_ddf = (
            self._ddf_as_mapping(inverse_ddf)
            .sample(data_format=DataFormat.voxel_displacements())
            .generate_values()
            .contiguous()
        )

        forward_ddfs = [
            forward_ddf if rank == self._process_within_group_rank else empty_like(forward_ddf)
            for rank in range(3)
        ]
        inverse_ddfs = [
            inverse_ddf if rank == self._process_within_group_rank else empty_like(inverse_ddf)
            for rank in range(3)
        ]
        handles = []
        for within_group_index, (broadcasted_forward_ddf, broadcasted_inverse_ddf) in enumerate(
            zip(forward_ddfs, inverse_ddfs)
        ):
            handles.append(
                broadcast(
                    broadcasted_forward_ddf,
                    src=self._training_group_indices[within_group_index],
                    async_op=True,
                    group=self._training_group,
                )
            )
            handles.append(
                broadcast(
                    broadcasted_inverse_ddf,
                    src=self._training_group_indices[within_group_index],
                    async_op=True,
                    group=self._training_group,
                )
            )
        logger.info("Rank %s waiting for DDFs", self._process_rank)
        for handle in handles:
            handle.wait()
        logger.info("Rank %s received all DDFs", self._process_rank)
        loss, loss_dict = self._compute_non_parallel_losses(
            forward_ddf_1=forward_ddfs[0],
            inverse_ddf_1=inverse_ddfs[0],
            forward_ddf_2=forward_ddfs[1],
            inverse_ddf_2=inverse_ddfs[1],
            forward_ddf_3=forward_ddfs[2],
            inverse_ddf_3=inverse_ddfs[2],
            mask_1=mask_1,
            mask_2=mask_2,
            mask_3=mask_3,
            loss=loss,
            loss_dict=loss_dict,
        )
        return loss, loss_dict

    def _ddf_as_mapping(self, ddf: Tensor, mask: Tensor | None = None) -> GridComposableMapping:
        mapping = samplable_volume(
            ddf,
            mask=mask,
            coordinate_system=self._model.transformation_coordinate_system,
            sampler=LinearInterpolator(),
            data_format=DataFormat.voxel_displacements(),
        )
        return mapping

    def _compute_directional_consistency_loss(
        self, mappings: Sequence[GridComposableMapping]
    ) -> Tensor:
        cycle: GridComposableMapping = Identity().assign_coordinates(
            self._model.transformation_coordinate_system
        )
        for mapping in mappings:
            cycle = cycle @ mapping
        cycle_ddf, cycle_mask = cycle.sample(DataFormat.world_displacements()).generate()
        consistency_loss_volume = (cycle_ddf.square() * cycle_mask).sum(dim=1)
        consistency_loss = consistency_loss_volume.mean()
        return consistency_loss

    def _compute_consistency_loss(
        self,
        mappings: Sequence[MappingPair],
    ) -> Tensor:
        forward_consistency_loss = self._compute_directional_consistency_loss(
            [mapping.forward_mapping for mapping in mappings],
        )
        inverse_consistency_loss = self._compute_directional_consistency_loss(
            [mapping.inverse_mapping for mapping in reversed(mappings)],
        )
        consistency_loss = (forward_consistency_loss + inverse_consistency_loss) / 2
        # Multiply the consistency loss gradients by 3 to account for the fact
        # that the gradients are averaged by DDP. This way the gradients are
        # consistent with the (non-existent) non-distributed case.
        return multiply_backward(consistency_loss, 3.0)

    def _compute_non_parallel_losses(
        self,
        forward_ddf_1: Tensor,
        inverse_ddf_1: Tensor,
        forward_ddf_2: Tensor,
        inverse_ddf_2: Tensor,
        forward_ddf_3: Tensor,
        inverse_ddf_3: Tensor,
        mask_1: Tensor | None,
        mask_2: Tensor | None,
        mask_3: Tensor | None,
        loss: Tensor,
        loss_dict: dict[str, float],
    ) -> tuple[Tensor, dict[str, float]]:
        final_mappings = [
            MappingPair(
                forward_mapping=self._ddf_as_mapping(forward_ddf_1, mask=mask_2),
                inverse_mapping=self._ddf_as_mapping(inverse_ddf_1, mask=mask_1),
            ),
            MappingPair(
                forward_mapping=self._ddf_as_mapping(forward_ddf_2, mask=mask_3),
                inverse_mapping=self._ddf_as_mapping(inverse_ddf_2, mask=mask_2),
            ),
            MappingPair(
                forward_mapping=self._ddf_as_mapping(forward_ddf_3, mask=mask_1),
                inverse_mapping=self._ddf_as_mapping(inverse_ddf_3, mask=mask_3),
            ),
        ]

        consistency_loss = self._compute_consistency_loss(
            final_mappings,
        )

        loss_dict = loss_dict | {
            "consistency": consistency_loss.item(),
        }
        loss = loss + self._get_consistency_loss_weight() * consistency_loss
        loss_dict["loss"] = loss.item()

        return loss, loss_dict
