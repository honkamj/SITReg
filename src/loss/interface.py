"""Interface to loss functions"""

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    overload,
)

from composable_mapping import GridComposableMapping, MappableTensor
from torch import Tensor

from util.import_util import import_object
from util.optional import optional_add


class ISimilarityLoss:
    """Similarity loss interface"""

    @abstractmethod
    def __call__(
        self,
        image_1: MappableTensor,
        image_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        pass


class ISegmentationLoss:
    """Segmentation loss interface"""

    @abstractmethod
    def __call__(
        self,
        seg_1: MappableTensor,
        seg_2: MappableTensor,
        params: Mapping[str, Any] | None = None,
    ) -> Tensor:
        pass


class IRegularityLoss:
    """Regularization loss interface"""

    @abstractmethod
    def __call__(
        self,
        mapping: GridComposableMapping,
        params: Mapping[str, Any] | None = None,
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


@overload
def get_combined_loss(
    losses: Sequence[LossDefinition[P]],
    *,
    prefix: str = "",
    previous_loss: Optional[Tensor] = None,
    previous_loss_dict: Optional[Mapping[str, float]] = None,
    weight: float = 1.0,
    default_kwargs_per_name: Mapping[str, Mapping[str, Any]] | None = None,
    tensor_loss_dict: Literal[False] = False,
) -> Callable[P, tuple[Tensor, dict[str, float]]]: ...


@overload
def get_combined_loss(
    losses: Sequence[LossDefinition[P]],
    *,
    prefix: str = "",
    previous_loss: Optional[Tensor] = None,
    previous_loss_dict: Optional[Mapping[str, float]] = None,
    weight: float = 1.0,
    default_kwargs_per_name: Mapping[str, Mapping[str, Any]] | None = None,
    tensor_loss_dict: Literal[True],
) -> Callable[P, tuple[Tensor, dict[str, Tensor]]]: ...


@overload
def get_combined_loss(
    losses: Sequence[LossDefinition[P]],
    *,
    prefix: str = "",
    previous_loss: Optional[Tensor] = None,
    previous_loss_dict: Optional[Mapping[str, float]] = None,
    weight: float = 1.0,
    default_kwargs_per_name: Mapping[str, Mapping[str, Any]] | None = None,
    tensor_loss_dict: bool = ...,
) -> Callable[P, tuple[Tensor, dict[str, float] | dict[str, Tensor]]]: ...


def get_combined_loss(
    losses: Sequence[LossDefinition[P]],
    *,
    prefix: str = "",
    previous_loss: Optional[Tensor] = None,
    previous_loss_dict: Optional[Mapping[str, float] | Mapping[str, Tensor]] = None,
    weight: float = 1.0,
    default_kwargs_per_name: Mapping[str, Mapping[str, Any]] | None = None,
    tensor_loss_dict: bool = False,
) -> Callable[P, tuple[Tensor, dict[str, float] | dict[str, Tensor]]]:
    """Compute multiple losses

    Returns callable with same parameters as the underlying losses.
    Callable returns the combined loss and updated dictionary of individual
    loss values as floats.
    """
    if default_kwargs_per_name is None:
        default_kwargs_per_name = {}

    def _compute_func(*args, **kwargs):
        combined_loss_dict: dict[str, float | Tensor] = (
            {"loss": float("nan")} if previous_loss_dict is None else dict(previous_loss_dict)
        )
        combined_loss: Optional[Tensor] = previous_loss
        for loss_definition in losses:
            default_kwargs = default_kwargs_per_name.get(loss_definition.name, {})
            loss_value = loss_definition.loss(*args, **kwargs | default_kwargs)
            combined_loss_dict[f"{prefix}{loss_definition.name}"] = (
                loss_value.detach() if tensor_loss_dict else loss_value.item()
            )
            combined_loss = optional_add(
                combined_loss, loss_value * loss_definition.weight * weight
            )
        assert combined_loss is not None
        combined_loss_dict["loss"] = (
            combined_loss.detach() if tensor_loss_dict else combined_loss.item()
        )
        return combined_loss, combined_loss_dict

    return _compute_func


def average_tensor_loss_dict(tensor_loss_dict: Mapping[str, Tensor]) -> dict[str, float]:
    """Convert tensor loss dictionary to float dictionary"""
    return {name: value.mean().item() for name, value in tensor_loss_dict.items()}


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


def create_segmentation_loss(segmentation_loss_config: Mapping[str, Any]) -> ISegmentationLoss:
    """Create segmentation loss"""
    loss_class = import_object(f'loss.segmentation.{segmentation_loss_config["type"]}')
    return loss_class(**segmentation_loss_config.get("args", {}))


def create_regularity_loss(regularity_loss_config: Mapping[str, Any]) -> IRegularityLoss:
    """Create regularization loss"""
    loss_class = import_object(f'loss.regularity.{regularity_loss_config["type"]}')
    return loss_class(**regularity_loss_config.get("args", {}))
