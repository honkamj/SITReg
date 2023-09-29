"""Metric related utils"""

from abc import abstractmethod
from json import dump as json_dump
from json import load as json_load
from os.path import isfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from numpy.ma import masked_invalid as np_masked_invalid
from torch import device as torch_device
from torch import empty, get_default_dtype, tensor
from torch.distributed import recv as distributed_recv
from torch.distributed import send as distributed_send


class LossAverager:
    """Utility class for averaging over individual loss outputs"""

    def __init__(
        self,
        displayed_metrics: Optional[Iterable[str]] = None,
        custom_mass_functions: Optional[Mapping[str, Callable[[float, float], float]]] = None,
    ) -> None:
        self._masses: dict[str, float] = {}
        self._loss_sums: dict[str, float] = {}
        self._displayed_metrics = displayed_metrics
        self._mass_functions: Mapping[str, Callable[[float, float], float]] = (
            {} if custom_mass_functions is None else custom_mass_functions
        )

    def count(self, losses: Mapping[str, float | tuple[float, float]]) -> None:
        """Add loss value to average

        Args:
            losses: If dictionary value is tuple of two floats, the other float is
                mass of that loss addition.
        """
        for loss_name, loss in losses.items():
            if isinstance(loss, tuple):
                loss_value, loss_mass = loss
            else:
                loss_value, loss_mass = loss, 1.0
            self._masses[loss_name] = self._masses.get(loss_name, 0) + loss_mass
            self._loss_sums[loss_name] = self._loss_sums.get(loss_name, 0.0) + loss_value

    def send(self, device: torch_device, target_training_process_rank: int) -> None:
        """Send loss data to other processes

        This is intended to be used together with the receive -method."""
        losses = tensor(
            [
                (self._loss_sums[loss_name], self._masses[loss_name])
                for loss_name in sorted(self._loss_sums.keys())
            ],
            device=device,
            dtype=get_default_dtype(),
        )
        distributed_send(losses, dst=target_training_process_rank)

    def receive(self, device: torch_device, source_training_process_rank: int) -> None:
        """Receive loss data from other processes"""
        receiving_tensor = empty(
            (len(self._loss_sums), 2), dtype=get_default_dtype(), device=device
        )
        distributed_recv(receiving_tensor, src=source_training_process_rank)
        self.count(
            {
                loss_name: (receiving_tensor[index][0].item(), receiving_tensor[index][1].item())
                for index, loss_name in enumerate(sorted(self._loss_sums.keys()))
            }
        )

    def save_to_json(self, epoch: int, filename: str, postfix: str = "") -> None:
        """Save loss to json-file"""
        if isfile(filename):
            with open(filename, mode="r", encoding="UTF-8") as loss_file_read:
                loss_dict = json_load(loss_file_read)
        else:
            loss_dict = {}
        losses = self._get_losses()
        losses = {f"{loss_name}{postfix}": loss_value for loss_name, loss_value in losses.items()}
        loss_dict[str(epoch + 1)] = losses
        with open(filename, mode="w", encoding="UTF-8") as loss_file_write:
            json_dump(loss_dict, loss_file_write, indent=4)

    def _get_losses(self) -> Mapping[str, float]:
        return {
            loss_name: self._get_mass_function(loss_name)(loss_sum, self._masses[loss_name])
            for loss_name, loss_sum in self._loss_sums.items()
        }

    def __repr__(self) -> str:
        return self.losses_as_string(self._get_losses())

    def _get_mass_function(self, loss_name: str) -> Callable[[float, float], float]:
        return self._mass_functions.get(loss_name, self._average)

    def _get_losses_to_display(self, available_losses: Iterable[str]) -> Iterable[str]:
        return available_losses if self._displayed_metrics is None else self._displayed_metrics

    def losses_as_string(self, losses: Mapping[str, float | tuple[float, float]]) -> str:
        """Get loss dictionary as string"""
        values: dict[str, float] = {}
        for loss_name, loss in losses.items():
            if isinstance(loss, tuple):
                value, mass = loss
            else:
                value, mass = loss, 1.0
            values[loss_name] = self._get_mass_function(loss_name)(value, mass)
        return ", ".join(
            f"{loss_name}={values[loss_name]:.4}"
            for loss_name in self._get_losses_to_display(losses.keys())
            if loss_name in losses
        )

    @staticmethod
    def _average(loss_sum: float, mass: float) -> float:
        if mass == 0:
            return float("nan")
        return loss_sum / mass


class ISummarizer:
    """Summarizer for summarizing metrics"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the summary"""

    @abstractmethod
    def __call__(self, metrics: Sequence[Any]) -> float:
        """Summarize metrics"""


class MetricsGatherer:
    """Utility class for gathering metrics from individual items and then
    calculating statistics from those

    Arguments:
        summarizers: Summarizers for summarizing metrics. Key None corresponds
            to default summarizers.
    """

    def __init__(self, summarizers: Mapping[str | None, Iterable[ISummarizer]]) -> None:
        self._metric_storage: dict[str, list[Any]] = {}
        self._summarizers = summarizers

    def count(self, metrics: Mapping[str, Any]) -> None:
        """Add metric values"""
        for metric_name, loss in metrics.items():
            if metric_name not in self._metric_storage:
                self._metric_storage[metric_name] = []
            self._metric_storage[metric_name].append(loss)

    def save_to_json(self, epoch_name: str, filename: str, prefix: str = "") -> None:
        """Save loss to json-file"""
        if isfile(filename):
            with open(filename, mode="r", encoding="UTF-8") as loss_file_read:
                metrics_dict = json_load(loss_file_read)
        else:
            metrics_dict = {}
        metrics = {
            f"{prefix}{metric_name}": summarized_metrics
            for metric_name, summarized_metrics in self._get_summarized_metrics().items()
        }
        metrics_dict[epoch_name] = metrics
        with open(filename, mode="w", encoding="UTF-8") as metric_file_write:
            json_dump(metrics_dict, metric_file_write, indent=4)

    def _get_summarizers(self, metric_name: str) -> Iterable[ISummarizer]:
        if metric_name in self._summarizers:
            return self._summarizers[metric_name]
        return self._summarizers[None]

    def _get_summarized_metrics(self) -> dict[str, dict[str, float]]:
        return {
            metric_name: {
                summarizer.name: summarizer(metrics)
                for summarizer in self._get_summarizers(metric_name)
            }
            for metric_name, metrics in self._metric_storage.items()
        }

    def __repr__(self) -> str:
        return ", ".join(
            f"{metric_name}={summarized_metrics:.4}"
            for metric_name, summarized_metrics in self._get_summarized_metrics().items()
        )


class MeanSummarizer(ISummarizer):
    """Mean summarizer"""

    def __init__(self, custom_name: str = None) -> None:
        self._name = "mean" if custom_name is None else custom_name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, metrics: Sequence[Any]) -> float:
        return float(np_masked_invalid(list(metrics)).mean())


class StdSummarizer(ISummarizer):
    """Std summarizer"""

    def __init__(self, custom_name: str = None) -> None:
        self._name = "std" if custom_name is None else custom_name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, metrics: Sequence[Any]) -> float:
        return float(np_masked_invalid(list(metrics)).std())


class LastSummarizer(ISummarizer):
    """Summarizer which returns the last metric value"""

    def __init__(self, custom_name: str = None) -> None:
        self._name = "last" if custom_name is None else custom_name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, metrics: Sequence[Any]) -> Any:
        return metrics[-1]
