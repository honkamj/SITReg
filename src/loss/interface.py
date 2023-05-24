"""Interface to loss functions"""

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
)

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from algorithm.composable_mapping.interface import (
    IComposableMapping,
    IMaskedTensor,
    VoxelCoordinateSystem,
)
from util.import_util import import_object
from util.optional import optional_add


class ISimilarityLoss:
    """Similarity loss interface"""

    @abstractmethod
    def __call__(
        self,
        image_1: IMaskedTensor,
        image_2: IMaskedTensor,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> Tensor:
        pass


class IRegularityLoss:
    """Regularization loss interface"""

    @abstractmethod
    def __call__(
        self,
        mapping: IComposableMapping,
        coordinate_system: VoxelCoordinateSystem,
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> Tensor:
        pass


P = ParamSpec("P")


class Loss(Protocol[P]):
    """Loss type"""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Tensor:
        pass


T = TypeVar("T", bound=Loss)


class LossDefinition(Generic[P]):
    """Loss definition"""

    def __init__(self, loss: Loss[P], weight: float, name: str) -> None:
        self.loss: Loss[P] = loss
        self.weight = weight
        self.name = name


def get_combined_loss(
    losses: Sequence[LossDefinition[P]],
    prefix: str = "",
    previous_loss: Optional[Tensor] = None,
    previous_loss_dict: Optional[Mapping[str, float]] = None,
    weight: float = 1.0,
) -> Callable[P, tuple[Tensor, dict[str, float]]]:
    """Compute multiple losses

    Returns callable with same parameters as the underlying losses.
    Callable returns dictionary with the combined loss at key "loss".
    Other dictionary items are detached from the computational graph.
    """

    def _compute_func(*args, **kwargs):
        combined_loss_dict: dict[str, float] = (
            {"loss": float('nan')} if previous_loss_dict is None else dict(previous_loss_dict)
        )
        combined_loss: Optional[Tensor] = previous_loss
        for loss_definition in losses:
            loss_value = loss_definition.loss(*args, **kwargs)
            combined_loss_dict[f"{prefix}{loss_definition.name}"] = loss_value.item()
            combined_loss = optional_add(
                combined_loss, loss_value * loss_definition.weight * weight
            )
        assert combined_loss is not None
        combined_loss_dict["loss"] = combined_loss.item()
        return combined_loss, combined_loss_dict

    return _compute_func


def create_losses(
    loss_configs: Sequence[Mapping[str, Any]],
    loss_init_func: Callable[[Mapping[str, Any]], Loss[P]],
) -> list[LossDefinition[P]]:
    """Create multiple losses"""
    loss_definitions: list[LossDefinition[P]] = []
    for loss_config in loss_configs:
        if loss_config["name"] == "loss":
            raise ValueError('Name "loss" is reserved for combined loss.')
        loss_definitions.append(
            LossDefinition(
                name=loss_config["name"],
                weight=loss_config.get("weight", 1.0),
                loss=loss_init_func(loss_config),
            )
        )
    return loss_definitions


def create_similarity_loss(similarity_loss_config: Mapping[str, Any]) -> ISimilarityLoss:
    """Create similarity loss"""
    loss_class = import_object(f'loss.similarity.{similarity_loss_config["type"]}')
    return loss_class(**similarity_loss_config.get("args", {}))


def create_regularity_loss(regularity_loss_config: Mapping[str, Any]) -> IRegularityLoss:
    """Create regularization loss"""
    loss_class = import_object(f'loss.regularity.{regularity_loss_config["type"]}')
    return loss_class(**regularity_loss_config.get("args", {}))
