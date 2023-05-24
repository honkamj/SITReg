"""Logging related utility functions"""


from logging import NOTSET, FileHandler, Formatter, StreamHandler, getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue
from os import environ


LEVEL_NAMES_MAPPING = {
    "CRITICAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
}


def _get_file_loglevel() -> int:
    return LEVEL_NAMES_MAPPING[environ.get("LOGLEVEL", "INFO")]


def _get_output_stream_loglevel() -> int:
    return LEVEL_NAMES_MAPPING[environ.get("OUTPUT_STREAM_LOGLEVEL", "WARNING")]


def configure_logging(log_output_file_path: str) -> None:
    """Configure logging"""
    formatter = Formatter("%(asctime)s - %(processName)s - %(levelname)s  - %(name)s - %(message)s")
    file_logging_handler = FileHandler(log_output_file_path, encoding='utf-8')
    file_logging_handler.setLevel(_get_file_loglevel())
    file_logging_handler.setFormatter(
        formatter
    )
    stream_logging_handler = StreamHandler()
    stream_logging_handler.setLevel(_get_output_stream_loglevel())
    stream_logging_handler.setFormatter(formatter)
    root_logger = getLogger()
    root_logger.addHandler(file_logging_handler)
    root_logger.addHandler(stream_logging_handler)
    root_logger.setLevel(NOTSET)


def configure_logging_for_subprocess(logging_queue: Queue) -> None:
    """Configure logging for subprocess"""
    root_logger = getLogger()
    root_logger.addHandler(QueueHandler(logging_queue))
    root_logger.setLevel(
        min(
            _get_file_loglevel(),
            _get_output_stream_loglevel()
        )
    )
