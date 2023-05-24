"""Git related utilities"""

from git import Repo  # type: ignore


def get_commit_hash() -> str:
    """Return current commit hash of the git HEAD"""
    return Repo(search_parent_directories=True).head.object.hexsha
