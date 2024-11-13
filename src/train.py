"""Training script"""

from argparse import ArgumentParser
from json import dump as json_dump
from logging import getLogger
from os import environ, makedirs
from os.path import join
from typing import Any, Iterable, Mapping, Protocol, Sequence, Sized

from torch import cuda
from torch import device as torch_device
from torch import set_default_dtype
from torch.distributed import Backend, destroy_process_group, init_process_group
from tqdm import tqdm  # type: ignore

from application.interface import (
    ITrainingDefinition,
    TrainingDefinitionArgs,
    create_training_definition,
)
from data.interface import TrainingDataLoaderArgs, create_training_data_loader
from scripts.file_names import (
    find_largest_epoch,
    loss_history_file_name,
    model_state_file_name,
    optimizer_state_file_name,
)
from util.checkpoint import load_states, save_states
from util.device import get_device_name
from util.git import get_commit_hash
from util.import_util import import_object
from util.json import load_json
from util.logging import configure_logging
from util.metrics import LossAverager

logger = getLogger(__name__)


class SizedIterable(Sized, Iterable, Protocol):  # pylint: disable=abstract-method
    """Sized iterable"""


def _save_states(
    training_definition: ITrainingDefinition, epoch: int, target_dir: str, dump: bool = False
) -> None:
    makedirs(target_dir, exist_ok=True)
    save_states(
        objects=training_definition.get_optimizers(),
        checkpoint_file_path=join(target_dir, optimizer_state_file_name(epoch, dump)),
    )
    save_states(
        objects=training_definition.get_modules(),
        checkpoint_file_path=join(target_dir, model_state_file_name(epoch, dump)),
    )


def _load_states(
    training_definition: ITrainingDefinition,
    epoch: int,
    load_optimizer_state: bool,
    target_dir: str,
    device: torch_device,
) -> None:
    if load_optimizer_state:
        load_states(
            objects=training_definition.get_optimizers(),
            checkpoint_file_path=join(target_dir, optimizer_state_file_name(epoch)),
            device=device,
        )
    load_states(
        objects=training_definition.get_modules(),
        checkpoint_file_path=join(target_dir, model_state_file_name(epoch)),
        device=device,
    )


def _init_process_group(
    local_rank: int,
    devices: Sequence[Sequence[torch_device]],
) -> None:
    first_device = devices[local_rank][0]
    init_process_group((Backend.GLOO if first_device.type == "cpu" else Backend.NCCL))
    if first_device.type == "cuda":
        cuda.set_device(devices[local_rank][0])


def _truncate(string: str, width: int):
    return string[: width - 3] + "..." if len(string) > width else string


def _train(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    config: Mapping[str, Any],
    target_dir: str,
    data_root: str,
    devices: Sequence[Sequence[torch_device]],
    continue_from_epoch: int | None = None,
    load_optimizer_state: bool = True,
    num_workers: int = 1,
) -> None:
    is_main_process = rank == 0
    logger.info("Starting training")
    process_devices = devices[local_rank]
    logger.info(
        "Using devices: %s", ", ".join([get_device_name(device) for device in process_devices])
    )
    training_definition = create_training_definition(
        config,
        args=TrainingDefinitionArgs(
            devices=process_devices,
            training_process_rank=rank,
            training_process_local_rank=local_rank,
            n_training_processes=world_size,
            n_local_training_processes=local_world_size,
        ),
    )
    training_data_loader = create_training_data_loader(
        config,
        args=TrainingDataLoaderArgs(
            data_root=data_root,
            num_workers=num_workers,
            training_process_rank=rank,
            training_process_local_rank=local_rank,
            n_training_processes=world_size,
            n_local_training_processes=local_world_size,
        ),
    )
    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        _load_states(
            training_definition=training_definition,
            epoch=continue_from_epoch,
            load_optimizer_state=load_optimizer_state,
            target_dir=target_dir,
            device=process_devices[0],
        )
        initial_epoch = continue_from_epoch + 1
        logger.info("Continuing training from epoch %d", initial_epoch + 1)
    if training_data_loader.generate_new_variant is not None:
        for epoch in range(initial_epoch):
            training_data_loader.generate_new_variant()
    if is_main_process:
        epoch_iterable: Iterable = tqdm(
            range(initial_epoch, training_definition.n_epochs),
            unit="epoch",
            initial=initial_epoch,
            total=training_definition.n_epochs,
        )
    else:
        epoch_iterable = range(initial_epoch, training_definition.n_epochs)

    for epoch in epoch_iterable:
        logger.info("Start of epoch %d", epoch + 1)
        loss_averager = LossAverager(
            displayed_metrics=training_definition.displayed_metrics(),
            custom_mass_functions=training_definition.get_custom_mass_functions(),
        )
        data_tqdm: tqdm | None = None
        if is_main_process:
            data_tqdm = tqdm(training_data_loader.data_loader, leave=False)
            data_iterable: SizedIterable = data_tqdm
        else:
            data_iterable = training_data_loader.data_loader
        try:
            n_steps = len(data_iterable)
            training_definition.start_of_epoch(epoch, n_steps)
            for step, batch in enumerate(data_iterable):
                loss_dict = training_definition.update_weights(batch)
                loss_averager.count(loss_dict)
                if data_tqdm is not None:
                    data_tqdm.set_description(_truncate(str(loss_averager), 200))
                logger.info(
                    "Epoch %d / %d, step  %d / %d, epoch average losses: %s, batch losses: %s",
                    epoch + 1,
                    training_definition.n_epochs,
                    step + 1,
                    n_steps,
                    loss_averager,
                    loss_averager.losses_as_string(loss_dict),
                )
        except KeyboardInterrupt:
            training_definition.before_save(saving_process_rank=0)
            if is_main_process:
                _save_states(training_definition, epoch, target_dir, dump=True)
            destroy_process_group()
            return
        training_definition.before_save(saving_process_rank=0)
        if is_main_process:
            _save_states(training_definition, epoch, target_dir)
            logger.info("Receiving loss information")
            for source_training_process_rank in range(1, world_size):
                loss_averager.receive(
                    device=process_devices[0],
                    source_training_process_rank=source_training_process_rank,
                )
                logger.info(
                    "Received loss information from the training process with rank %d",
                    source_training_process_rank,
                )
            loss_averager.save_to_json(
                epoch=epoch, filename=join(target_dir, loss_history_file_name())
            )
        else:
            logger.info("Sending loss information to the training process with rank 0")
            loss_averager.send(device=process_devices[0], target_training_process_rank=0)
        logger.info("End of epoch %d", epoch + 1)
        if training_data_loader.generate_new_variant is not None:
            training_data_loader.generate_new_variant()


def _main() -> None:
    """Parse arguments for training and train the model"""
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, required=False)
    parser.add_argument("--training-root", help="Path to output root", type=str, required=True)
    parser.add_argument("--data-root", help="Path to data root", type=str, required=True)
    parser.add_argument(
        "--num-workers", help="Number of workers for data generation", type=int, required=True
    )
    parser.add_argument(
        "--model-name", help="Model name used in saving the model", type=str, required=True
    )
    parser.add_argument(
        "--continue-from-epoch",
        help="Training is continued from epoch after this",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--do-not-load-optimizer-state",
        help="Training is continued without loading optimizer state",
        action="store_true",
    )

    parser.add_argument(
        "--devices",
        help=(
            "Names of the devices to use for each training process. "
            "Include the argument multiple times to specify devices for "
            "multiple processes."
        ),
        type=str,
        nargs="+",
        action="append",
    )
    args = parser.parse_args()
    target_dir = join(args.training_root, args.model_name)
    data_root = args.data_root
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    config = load_json(config_path)
    set_default_dtype(import_object(config.get("dtype", "torch.float32")))
    makedirs(target_dir, exist_ok=True)
    if args.continue_from_epoch is None:
        continue_from_epoch = find_largest_epoch(
            target_dir, require_optimizer=not args.do_not_load_optimizer_state
        )
    else:
        continue_from_epoch = int(args.continue_from_epoch) - 1
    devices = [
        [torch_device(device_name) for device_name in device_names] for device_names in args.devices
    ]
    with open(
        join(target_dir, "training_config.json"), mode="w", encoding="UTF-8"
    ) as config_copy_file:
        config["commit"] = get_commit_hash()
        json_dump(config, config_copy_file, indent=4)
    local_rank = int(environ.get("LOCAL_RANK", 0))
    rank = int(environ.get("RANK", 0))
    world_size = int(environ.get("WORLD_SIZE", 1))
    log_path = join(target_dir, f"training_log_{rank}.out")
    if world_size > 1:
        _init_process_group(local_rank, devices)
    configure_logging(log_path)
    _train(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=int(environ.get("LOCAL_WORLD_SIZE", 1)),
        config=config,
        target_dir=target_dir,
        data_root=data_root,
        devices=devices,
        continue_from_epoch=continue_from_epoch,
        load_optimizer_state=not args.do_not_load_optimizer_state,
        num_workers=args.num_workers,
    )
    if world_size > 1:
        destroy_process_group()


if __name__ == "__main__":
    _main()
