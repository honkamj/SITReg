"""Utils for extracting files"""

from gzip import GzipFile
from os import makedirs, remove
from os.path import basename, dirname, join, splitext
from tarfile import open as open_tar
from zipfile import ZipFile


def untar(file_path: str, extract_to_same_dir: bool = False, remove_after: bool = True) -> None:
    """Untar file"""
    if extract_to_same_dir:
        target_dir = dirname(file_path)
    else:
        target_dir = _get_target_path(file_path)
    makedirs(target_dir, exist_ok=True)
    tar = open_tar(file_path)
    tar.extractall(target_dir)
    if remove_after:
        remove(file_path)


def unzip(file_path: str, extract_to_same_dir: bool = False, remove_after: bool = True) -> None:
    """Unzip file"""
    if extract_to_same_dir:
        target_dir = dirname(file_path)
    else:
        target_dir = _get_target_path(file_path)
    makedirs(target_dir, exist_ok=True)
    with ZipFile(file_path, 'r') as zip_file:
        zip_file.extractall(target_dir)
    if remove_after:
        remove(file_path)


def ungzip(file_path: str, remove_after: bool = True) -> None:
    """Ungzip file"""
    target_file = _get_target_path(file_path)
    with GzipFile(file_path, 'r') as gzip_file:
        with open(target_file, 'wb') as ungzipped_file:
            ungzipped_file.write(gzip_file.read())
    if remove_after:
        remove(file_path)


def _get_target_path(file_path: str) -> str:
    return join(dirname(file_path), splitext(basename(file_path))[0])
