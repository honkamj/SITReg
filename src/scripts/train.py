"""Training script"""

from argparse import ArgumentParser
from json import dump as json_dump
from logging import getLogger
from multiprocessing import Queue, get_context
from os import environ, makedirs
from os.path import join
from threading import Thread
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Sized

from torch import device as torch_device
from torch.distributed import init_process_group, destroy_process_group, Backend
from torch.multiprocessing import spawn
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
from util.json import load_json
from util.logging import configure_logging_for_subprocess, configure_logging
from util.metrics import LossAverager

logger = getLogger(__name__)


class SizedIterable(Sized, Iterable, Protocol): # pylint: disable=abstract-method
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
    training_definition: ITrainingDefinition, epoch: int, target_dir: str, device: torch_device
) -> None:
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
    training_process_rank: int, devices: Sequence[torch_device], logging_queue: Queue
) -> None:
    init_process_group(
        Backend.GLOO if all(device.type == "cpu" for device in devices) else Backend.NCCL,
        rank=training_process_rank,
        world_size=len(devices),
    )
    configure_logging_for_subprocess(logging_queue)


def _truncate(string: str, width: int):
    return string[:width-3] + '...' if len(string) > width else string


def _train(
    training_process_rank: int,
    config: Mapping[str, Any],
    target_dir: str,
    data_root: str,
    devices: Sequence[torch_device],
    continue_from_epoch: int | None = None,
    num_workers: int = 1,
    logging_queue: Optional[Queue] = None,
) -> None:
    n_training_processes = len(devices)
    is_main_process = training_process_rank == 0
    if n_training_processes > 1:
        assert logging_queue is not None
        _init_process_group(training_process_rank, devices, logging_queue)
    logger.info("Starting training")
    device = devices[training_process_rank]
    logger.info("Using device %s", get_device_name(device))
    training_definition = create_training_definition(
        config,
        args=TrainingDefinitionArgs(
            device=device,
            training_process_rank=training_process_rank,
            n_training_processes=n_training_processes,
        ),
    )
    training_data_loader = create_training_data_loader(
        config,
        args=TrainingDataLoaderArgs(
            data_root=data_root,
            num_workers=num_workers,
            training_process_rank=training_process_rank,
            n_training_processes=n_training_processes,
        ),
    )
    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        _load_states(training_definition, continue_from_epoch, target_dir, device)
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
            for step, batch in enumerate(data_iterable):
                loss_dict = training_definition.update_weights(batch)
                loss_averager.count(loss_dict)
                if data_tqdm is not None:
                    data_tqdm.set_description(_truncate(str(loss_averager), 80))
                logger.info(
                    "Epoch %d / %d, step  %d / %d, epoch average losses: %s, "
                    "batch losses: %s",
                    epoch + 1,
                    training_definition.n_epochs,
                    step + 1,
                    n_steps,
                    loss_averager,
                    loss_averager.losses_as_string(loss_dict)
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
            for source_training_process_rank in range(1, n_training_processes):
                loss_averager.receive(
                    device=device, source_training_process_rank=source_training_process_rank
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
            loss_averager.send(device=device, target_training_process_rank=0)
        logger.info("End of epoch %d", epoch + 1)
        if training_data_loader.generate_new_variant is not None:
            training_data_loader.generate_new_variant()
    if n_training_processes > 1:
        destroy_process_group()


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
    parser.add_argument("--devices", help="Names of the devices to use", type=str, nargs="+")
    args = parser.parse_args()
    target_dir = join(args.training_root, args.model_name)
    data_root = args.data_root
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    config = load_json(config_path)
    makedirs(target_dir, exist_ok=True)
    if args.continue_from_epoch is None:
        continue_from_epoch = find_largest_epoch(target_dir, require_optimizer=True)
    else:
        continue_from_epoch = args.continue_from_epoch
    devices = [torch_device(device_name) for device_name in args.devices]
    with open(
        join(target_dir, "training_config.json"), mode="w", encoding="UTF-8"
    ) as config_copy_file:
        config["commit"] = get_commit_hash()
        json_dump(config, config_copy_file, indent=4)
    log_path = join(target_dir, "training_log.out")
    configure_logging(log_path)
    print(f'Log written to "{log_path}"')
    if len(devices) == 1:
        _train(
            training_process_rank=0,
            config=config,
            target_dir=target_dir,
            data_root=data_root,
            devices=devices,
            continue_from_epoch=continue_from_epoch,
            num_workers=args.num_workers,
        )
    else:
        environ["MASTER_ADDR"] = "localhost"
        environ["MASTER_PORT"] = "29500"
        multiprocessing_context = get_context("spawn")
        logging_queue: Queue = multiprocessing_context.Queue(-1)
        logging_listener = Thread(target=_logging_listener, args=(logging_queue,))
        logging_listener.start()
        try:
            spawn(
                _train,
                args=(
                    config,
                    target_dir,
                    data_root,
                    devices,
                    continue_from_epoch,
                    args.num_workers,
                    logging_queue,
                ),
                nprocs=len(devices),
                join=True,
            )
        except (KeyboardInterrupt, SystemExit, Exception) as exception:
            _exit_logging_listener(logging_listener, logging_queue)
            raise exception
        _exit_logging_listener(logging_listener, logging_queue)


def _exit_logging_listener(logging_listener: Thread, logging_queue: Queue) -> None:
    logging_queue.put_nowait(None)
    logging_listener.join()


def _logging_listener(logging_queue: Queue):
    while True:
        record = logging_queue.get(block=True, timeout=None)
        if record is None:
            break
        getLogger(record.name).handle(record)


if __name__ == "__main__":
    _main()
