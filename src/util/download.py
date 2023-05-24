"""Utils for downloading files"""


from os import makedirs
from os.path import dirname
from typing import Optional
from urllib.request import urlretrieve

from tqdm import tqdm  # type: ignore


def download(source_url: str, target_path: str, description: Optional[str] = None) -> None:
    """Download file from source_url to target_path"""
    makedirs(dirname(target_path), exist_ok=True)
    progress_bar = None
    previous_recieved = 0

    def _show_progress(block_num, block_size, total_size):
        nonlocal progress_bar, previous_recieved
        if progress_bar is None:
            progress_bar = tqdm(unit="B", total=total_size)
            if description is not None:
                progress_bar.set_description(description)
        downloaded = block_num * block_size
        if downloaded < total_size:
            progress_bar.update(downloaded - previous_recieved)
            previous_recieved = downloaded
        else:
            progress_bar.close()

    urlretrieve(source_url, target_path, _show_progress)
