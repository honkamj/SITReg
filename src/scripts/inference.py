"""Inference script"""

from argparse import ArgumentParser
from logging import getLogger
from multiprocessing import Queue, get_context
from os import makedirs, remove
from os.path import isfile, join
from random import shuffle
from threading import Thread
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence

from torch import device as torch_device
from torch import no_grad, set_default_dtype
from torch.multiprocessing import spawn
from tqdm import tqdm  # type: ignore

from application.interface import IInferenceDefinition, create_inference_definition
from data.interface import (
    IInferenceFactory,
    InferenceDataArgs,
    create_inference_data_factory,
)
from data.storage import (
    load_items_from_storages,
    save_items_to_storages,
    storages_exist,
)
from scripts.file_names import (
    case_metrics_file_name,
    find_largest_epoch,
    get_optional_epoch_as_string,
    inference_subfolder,
    metrics_file_name,
    model_state_file_name,
)
from util.checkpoint import load_states
from util.import_util import import_object
from util.json import load_json, save_json
from util.logging import configure_logging, configure_logging_for_subprocess
from util.metrics import MetricsGatherer

logger = getLogger(__name__)


class _InferenceArgs(NamedTuple):
    config: Mapping[str, Any]
    target_dir: str
    data_root: str
    inference_folder: str
    epoch: int | None
    division: str
    save_outputs: bool
    evaluate: bool
    skip_existing_evaluations: bool
    skip_existing_outputs: bool
    devices: Sequence[torch_device]
    preload_data: bool
    num_workers: int
    num_dummy_inferences: int
    shuffle_cases: bool
    instance_index: int
    n_instances: int


def _load_states(
    inference_definition: IInferenceDefinition, epoch: int, target_dir: str, device: torch_device
) -> None:
    load_states(
        objects=inference_definition.get_modules(),
        checkpoint_file_path=join(target_dir, model_state_file_name(epoch)),
        device=device,
    )


def _inference_for_index(
    inference_process_rank: int,
    case_index: int,
    inference_data_factory: IInferenceFactory,
    inference_definition: IInferenceDefinition,
    args: _InferenceArgs,
    load_model_state_if_not_loaded: Callable[[], None] | None,
) -> Mapping[str, Any]:
    metadata = inference_data_factory.get_metadata(case_index)
    logger.info(
        "Starting inference for input %s",
        metadata.inference_name,
    )
    case_inference_folder = join(
        args.target_dir,
        inference_subfolder(
            args.inference_folder, args.epoch, args.division, metadata.inference_name
        ),
    )
    makedirs(case_inference_folder, exist_ok=True)
    case_metrics_path = join(case_inference_folder, case_metrics_file_name())
    output_storages = inference_definition.get_output_storages(metadata)

    outputs: Mapping[str, Any] | None = None
    case_metrics: Mapping[str, Any] | None = None

    do_evaluation = args.evaluate and (
        not args.skip_existing_evaluations
        or (args.skip_existing_evaluations and not isfile(case_metrics_path))
    )
    do_inference = (
        do_evaluation or args.save_outputs or (not args.skip_existing_outputs and not args.evaluate)
    ) and (
        not args.skip_existing_outputs
        or (
            not storages_exist(
                target_folder=case_inference_folder, storages=output_storages.values()
            )
        )
    )
    if do_inference:
        if load_model_state_if_not_loaded is not None:
            load_model_state_if_not_loaded()
        case_data_iterable: Iterable[Any] = inference_data_factory.get_data_loader(
            case_index, num_workers=args.num_workers
        )
        if args.preload_data:
            case_data_iterable = list(case_data_iterable)
        with inference_definition.get_case_inference(metadata) as case_inferencer:
            with no_grad():
                for batch in case_data_iterable:
                    case_inferencer.infer(batch)
        outputs = case_inferencer.get_outputs()
        if args.save_outputs:
            logger.info("Saving inference outputs for input %s", metadata.inference_name)
            save_items_to_storages(
                target_folder=case_inference_folder, storages=output_storages, items=outputs
            )
    else:
        logger.info(
            "Skipped inference for input %s",
            metadata.inference_name,
        )
    if args.evaluate and isfile(case_metrics_path):
        case_metrics = load_json(case_metrics_path)
    else:
        case_metrics = {}
    if do_evaluation:
        logger.info(
            "Starting evaluation for input %s",
            metadata.inference_name,
        )
        device = args.devices[inference_process_rank]
        evaluator = inference_data_factory.get_evaluator(case_index, device=device)
        if outputs is None:
            outputs = load_items_from_storages(
                target_folder=case_inference_folder,
                storages=output_storages,
                device=device,
                only_names=evaluator.evaluation_inference_outputs,
            )
        case_metrics.update(evaluator(outputs))
        try:
            save_json(path=case_metrics_path, data=case_metrics)
        except KeyboardInterrupt:
            if isfile(case_metrics_path):
                remove(case_metrics_path)
            raise
    elif args.evaluate:
        logger.info(
            "Skipped evaluation for input %s",
            metadata.inference_name,
        )
    return case_metrics


class _LoadModelIfNotLoaded:
    def __init__(
        self,
        inference_definition: IInferenceDefinition,
        epoch: int,
        target_dir: str,
        device: torch_device,
    ):
        self._inference_definition = inference_definition
        self._epoch = epoch
        self._target_dir = target_dir
        self._device = device
        self._is_loaded = False

    def __call__(self):
        if not self._is_loaded:
            _load_states(
                inference_definition=self._inference_definition,
                epoch=self._epoch,
                target_dir=self._target_dir,
                device=self._device,
            )
            self._is_loaded = True


def _inference_process(
    inference_process_rank: int,
    inference_data_factory: IInferenceFactory,
    args: _InferenceArgs,
    logging_queue: Optional[Queue],
    case_index_queue: Queue,
    output_queue: Queue,
) -> None:
    use_multiprocessing = len(args.devices) > 1
    if use_multiprocessing:
        assert logging_queue is not None
        configure_logging_for_subprocess(logging_queue)
    device = args.devices[inference_process_rank]
    inference_definition = create_inference_definition(args.config, device)
    if args.epoch is not None:
        load_model_state_if_not_loaded: _LoadModelIfNotLoaded | None = _LoadModelIfNotLoaded(
            inference_definition=inference_definition,
            epoch=args.epoch,
            target_dir=args.target_dir,
            device=device,
        )
    else:
        load_model_state_if_not_loaded = None

    if args.num_dummy_inferences > 0:
        logger.info("Allocating memory")
        dummy_batch, dummy_metadata = inference_data_factory.generate_dummy_batch_and_metadata()
        with inference_definition.get_case_inference(dummy_metadata) as case_inferencer:
            with no_grad():
                for _ in range(args.num_dummy_inferences):
                    case_inferencer.infer(dummy_batch)
    while True:
        case_index: int | None = case_index_queue.get(block=True, timeout=None)
        if case_index is None:
            return
        output_queue.put(
            _inference_for_index(
                inference_process_rank=inference_process_rank,
                case_index=case_index,
                inference_data_factory=inference_data_factory,
                inference_definition=inference_definition,
                args=args,
                load_model_state_if_not_loaded=load_model_state_if_not_loaded,
            )
        )


def _inference(args: _InferenceArgs) -> None:
    logger.info(
        "Starting inference for epoch %s, division %s",
        get_optional_epoch_as_string(args.epoch),
        args.division,
    )
    n_inference_processes = len(args.devices)
    multiprocessing_context = get_context("spawn")
    case_index_queue = multiprocessing_context.Queue(-1)
    output_queue = multiprocessing_context.Queue(-1)
    inference_data_factory = create_inference_data_factory(
        config=args.config, args=InferenceDataArgs(data_root=args.data_root, division=args.division)
    )
    n_cases = len(inference_data_factory)
    case_indices = list(range(n_cases))
    case_indices = case_indices[args.instance_index :: args.n_instances]
    if args.shuffle_cases:
        shuffle(case_indices)
    evaluation_listening_args = (
        _EvaluationListeningArgs(
            inference_data_factory=inference_data_factory,
            epoch_name=get_optional_epoch_as_string(args.epoch),
            metrics_filename=join(
                args.target_dir, metrics_file_name(args.inference_folder, args.division)
            ),
        )
        if args.evaluate and args.n_instances == 1
        else None
    )
    inference_result_listener = Thread(
        target=_inference_results_listener,
        args=(
            output_queue,
            case_index_queue,
            case_indices[n_inference_processes:],
            n_cases,
            evaluation_listening_args,
        ),
    )
    inference_result_listener.start()
    for case_index in case_indices[:n_inference_processes]:
        case_index_queue.put(case_index)
    use_multiprocessing = n_inference_processes > 1
    try:
        if use_multiprocessing:
            logging_queue: Queue = multiprocessing_context.Queue(-1)
            logging_listener = Thread(target=_logging_listener, args=(logging_queue,))
            logging_listener.start()
            try:
                spawn(
                    _inference_process,
                    args=(
                        inference_data_factory,
                        args,
                        logging_queue,
                        case_index_queue,
                        output_queue,
                    ),
                    nprocs=n_inference_processes,
                    join=True,
                )
            except (KeyboardInterrupt, SystemExit, Exception) as exception:
                _exit_thread(logging_listener, logging_queue)
                raise exception
            _exit_thread(logging_listener, logging_queue)
        else:
            _inference_process(
                inference_process_rank=0,
                inference_data_factory=inference_data_factory,
                args=args,
                logging_queue=None,
                case_index_queue=case_index_queue,
                output_queue=output_queue,
            )
    except (KeyboardInterrupt, SystemExit, Exception) as exception:
        _exit_thread(inference_result_listener, output_queue)
        raise exception
    inference_result_listener.join()


def _exit_thread(thread: Thread, exit_queue: Queue) -> None:
    exit_queue.put_nowait(None)
    thread.join()


def _logging_listener(logging_queue: Queue):
    while True:
        record = logging_queue.get(block=True, timeout=None)
        if record is None:
            return
        getLogger(record.name).handle(record)


class _EvaluationListeningArgs(NamedTuple):
    inference_data_factory: IInferenceFactory
    epoch_name: str
    metrics_filename: str


def _inference_results_listener(
    output_queue: Queue,
    case_index_queue: Queue,
    case_indices: Sequence[int],
    n_cases: int,
    evaluation_listening_args: _EvaluationListeningArgs | None,
):
    case_indices_list = list(case_indices)
    if evaluation_listening_args is not None:
        metrics_gatherer = MetricsGatherer(
            summarizers=evaluation_listening_args.inference_data_factory.get_evaluator_summarizers()
        )
    for _ in tqdm(range(n_cases)):
        output = output_queue.get(block=True, timeout=None)
        if output is None:
            return
        if case_indices_list:
            case_index_queue.put(case_indices_list.pop(0))
        else:
            case_index_queue.put(None)
        if evaluation_listening_args is not None:
            metrics_gatherer.count(output)
    if evaluation_listening_args is not None:
        metrics_gatherer.save_to_json(
            epoch_name=evaluation_listening_args.epoch_name,
            filename=evaluation_listening_args.metrics_filename,
        )


def _main() -> None:
    """Parse arguments for training and train the model"""
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, required=False)
    parser.add_argument("--training-root", help="Path to output root", type=str, required=True)
    parser.add_argument("--data-root", help="Path to data root", type=str, required=True)
    parser.add_argument("--division", help="Dataset division", type=str, required=True)
    parser.add_argument(
        "--num-workers",
        help="Number of workers for data generation",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--model-name", help="Model name used in saving the model", type=str, required=True
    )
    parser.add_argument("--epoch", help="Epoch to do inference", type=str, nargs="*")
    parser.add_argument("--devices", help="Names of the devices to use", type=str, nargs="+")
    parser.add_argument(
        "--do-not-save-outputs", help="Do not save outputs to disk", action="store_true"
    )
    parser.add_argument("--shuffle-cases", help="Shuffle cases", action="store_true")
    parser.add_argument(
        "--evaluate", help="Perform evaluation of inference outputs", action="store_true"
    )
    parser.add_argument(
        "--skip-existing-evaluations", help="Skip existing evaluations", action="store_true"
    )
    parser.add_argument(
        "--skip-existing-outputs", help="Skip existing outputs", action="store_true"
    )
    parser.add_argument(
        "--preload-data",
        help=(
            "Load all input data of a case to memory before starting the inference. "
            "Useful in measuring inference time."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--inference-folder",
        help="Give custom name for inference folder",
        type=str,
        required=False,
        default="inference",
    )
    parser.add_argument(
        "--num-dummy-inferences",
        help=(
            "Number of dummy inferences to perform before actual inference to ensure "
            "correct inference time measurements"
        ),
        type=int,
        default=2,
    )
    parser.add_argument(
        "--instance-index",
        help=(
            "If running the inference using multiple instances, define the index of the instance "
            "with this parameter. No summary will be built if using more than one instance."
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n-instances",
        help=(
            "If running the inference using multiple instances, define the number of the instances "
            "with this parameter. No summary will be built if using more than one instance."
        ),
        type=int,
        default=1,
    )
    args = parser.parse_args()
    target_dir = join(args.training_root, args.model_name)
    makedirs(target_dir, exist_ok=True)
    data_root = args.data_root
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    config = load_json(config_path)
    set_default_dtype(import_object(config.get("dtype", "torch.float32")))
    if args.epoch is None:
        epoch_candidate = find_largest_epoch(target_dir, require_optimizer=False)
        epochs: list[int | None] = [epoch_candidate]
    else:
        epochs = []
        for entered_epoch in args.epoch:
            if entered_epoch == "best_epoch":
                with open(
                    join(target_dir, args.inference_folder, "best_epoch.txt"),
                    mode="r",
                    encoding="UTF-8",
                ) as epoch_file:
                    epochs.append(int(epoch_file.read().strip()) - 1)
            elif len(entered_epoch.split("-")) == 2:
                start_epoch_str, end_epoch_str = entered_epoch.split("-")
                epochs.extend(range(int(start_epoch_str) - 1, int(end_epoch_str)))
            else:
                epochs.append(int(entered_epoch) - 1)
    devices = [torch_device(device_name) for device_name in args.devices]
    log_path = join(target_dir, f"{args.inference_folder}_log.out")
    configure_logging(log_path)
    print(f'Log written to "{log_path}"')
    for epoch in epochs:
        _inference(
            _InferenceArgs(
                config=config,
                target_dir=target_dir,
                data_root=data_root,
                inference_folder=args.inference_folder,
                epoch=epoch,
                division=args.division,
                save_outputs=not args.do_not_save_outputs,
                evaluate=args.evaluate,
                skip_existing_evaluations=args.skip_existing_evaluations,
                skip_existing_outputs=args.skip_existing_outputs,
                devices=devices,
                preload_data=args.preload_data,
                num_workers=args.num_workers,
                num_dummy_inferences=args.num_dummy_inferences,
                shuffle_cases=args.shuffle_cases,
                instance_index=args.instance_index,
                n_instances=args.n_instances,
            )
        )


if __name__ == "__main__":
    _main()
