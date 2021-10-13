import os.path
from typing import Iterable, Optional

from tqdm import tqdm


def check_file_exists(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Provided file {file_path} doesn't exist.")

    return file_path


def progress_bar(
    itr: Iterable, title: Optional[str] = None, show: bool = True
) -> Iterable:
    if show:
        return tqdm(itr, title=title)
    else:
        return itr
