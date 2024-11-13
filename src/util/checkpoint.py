"""Model checkpointing related utils"""

from os import environ
from typing import Mapping, Union

from torch import device as torch_device
from torch import get_default_dtype, load, save
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def load_states(
    objects: Mapping[str, Union[Module, Optimizer, LRScheduler]],
    checkpoint_file_path: str,
    device: torch_device,
) -> None:
    """Load states from checkpoint"""
    checkpoint = load(checkpoint_file_path, map_location=device)
    for object_name, target_object in objects.items():
        if isinstance(target_object, Module):
            strict = environ.get("STRICT_STATE_DICT_LOADING", "true").lower() == "true"
            if object_name in checkpoint or strict:
                target_object.load_state_dict(checkpoint[object_name], strict=strict)
                target_object.to(get_default_dtype())
        else:
            target_object.load_state_dict(checkpoint[object_name])


def save_states(
    objects: Mapping[str, Union[Module, Optimizer, LRScheduler]], checkpoint_file_path: str
) -> None:
    """Save states to checkpoint"""
    dictionary_to_save = {}
    for object_name, target_object in objects.items():
        dictionary_to_save[object_name] = target_object.state_dict()
    save(dictionary_to_save, checkpoint_file_path)
