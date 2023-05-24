"""File name conventions"""

from os import listdir
from os.path import isdir, isfile, join
from re import match
from typing import Optional


def optimizer_state_file_name(
        epoch: int,
        dump: bool = False
    ) -> str:
    """Optimizer state file name"""
    if dump:
        return f'optimizer_epoch_{epoch + 1:03d}_dump.pt'
    return f'optimizer_epoch_{epoch + 1:03d}.pt'


def model_state_file_name(
        epoch: int,
        dump: bool = False
    ) -> str:
    """Model state file name"""
    if dump:
        return f'model_epoch_{epoch + 1:03d}_dump.pt'
    return f'model_epoch_{epoch + 1:03d}.pt'


def extract_model_state_epoch(state_file_name: str) -> Optional[int]:
    """Extract epoch from state file name"""
    epoch_string = match(r'model_epoch_([0-9]{2}[0-9]+)\.pt', state_file_name)
    if epoch_string is not None:
        return int(epoch_string.group(1)) - 1
    return None


def loss_history_file_name() -> str:
    """Loss history file name"""
    return 'loss_history.json'


def metrics_file_name(division: str) -> str:
    """Metrics file name"""
    return f'metrics_{division}.json'


def case_metrics_file_name() -> str:
    """Metrics file name for a single inference case"""
    return 'metrics.json'


def find_largest_epoch(
        training_directory: str,
        require_optimizer: bool = True
    ) -> Optional[int]:
    """Find largest epoch from directory"""
    largest_epoch: Optional[int] = None
    if not isdir(training_directory):
        return None
    for entry in listdir(training_directory):
        if isfile(join(training_directory, entry)):
            epoch = extract_model_state_epoch(entry)
            if (
                    epoch is not None and
                    (largest_epoch is None or epoch > largest_epoch) and
                    (
                        not require_optimizer or
                        isfile(join(training_directory, optimizer_state_file_name(epoch)))
                    )
                ):
                largest_epoch = epoch
    return largest_epoch


def inference_subfolder(
        epoch: int | None,
        division: str,
        case_name: str,
    ) -> str:
    """Inference folder"""
    if epoch is None:
        epoch_folder = 'no_epoch'
    else:
        epoch_folder = f'epoch_{epoch + 1:03d}'
    return join('inference', epoch_folder, division, case_name)


def get_optional_epoch_as_string(epoch: int | None) -> str:
    """Get optional epoch as string"""
    return "no_epoch" if epoch is None else str(epoch + 1)
