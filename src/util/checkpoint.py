"""Model checkpointing related utils"""


from os import environ
from typing import Mapping, Union

from torch import device as torch_device
from torch import load, save
from torch.nn import Module
from torch.optim import Optimizer


def load_states(
    objects: Mapping[str, Union[Module, Optimizer]], checkpoint_file_path: str, device: torch_device
) -> None:
    """Load states from checkpoint"""
    checkpoint = load(checkpoint_file_path, map_location=device)
    for object_name, target_object in objects.items():
        if isinstance(target_object, Module):
            strict = environ.get("STRICT_STATE_DICT_LOADING", "true").lower() == "true"
            target_object.load_state_dict(checkpoint[object_name], strict=strict)
        else:
            target_object.load_state_dict(checkpoint[object_name])


def save_states(objects: Mapping[str, Union[Module, Optimizer]], checkpoint_file_path: str) -> None:
    """Save states to checkpoint"""
    dictionary_to_save = {}
    for object_name, target_object in objects.items():
        dictionary_to_save[object_name] = target_object.state_dict()
    save(dictionary_to_save, checkpoint_file_path)
