"""Saver implementations"""

from os import listdir, makedirs, remove, rmdir
from os.path import isfile, join
from typing import Any, Iterable, Mapping, Optional, Sequence

from nibabel import Nifti1Image  # type: ignore
from nibabel import save as nib_save
from numpy import load as numpy_load
from numpy import ndarray
from numpy import save as numpy_save
from numpy import savez as numpy_savez
from torch import Tensor, as_tensor
from torch import device as torch_device
from torch import load, save

from .interface import IStorage, IStorageFactory


class NiftiStorageFactory(IStorageFactory):
    """Save nifti data to disk"""

    def __init__(self, affine: Optional[Tensor] = None) -> None:
        self._affine = affine

    def create(self, name: str) -> IStorage:
        return NiftiStorage(name=name, affine=self._affine)


class BaseStorage(IStorage):
    """Base storage implementation"""

    def __init__(self, target_file: str) -> None:
        self._target_file = target_file

    def _get_path(self, target_folder: str) -> str:
        return join(target_folder, self._target_file)

    def exists(self, target_folder: str) -> bool:
        return isfile(self._get_path(target_folder))

    def clear(self, target_folder: str) -> None:
        if self.exists(target_folder):
            remove(self._get_path(target_folder))


class NiftiNumpyStorage(BaseStorage):
    """Save nifti data to disk from numpy"""

    def __init__(self, name: str, affine: Optional[ndarray] = None) -> None:
        super().__init__(f"{name}.nii.gz")
        self._affine = affine

    def _get_data(self, item: Any) -> ndarray:
        data: ndarray = item
        if data.shape[0] != 1:
            raise ValueError("Only one channel data is supported")
        return data[0]

    def save(self, item: Any, target_folder: str) -> None:
        nifti_image = Nifti1Image(self._get_data(item), affine=self._affine)
        nib_save(nifti_image, self._get_path(target_folder))

    def load(self, target_folder: str, device: torch_device) -> Any:
        raise NotImplementedError("Nifti loading not implemented")


class NiftiStorage(NiftiNumpyStorage):
    """Save nifti data to disk from torch Tensor"""

    def __init__(self, name: str, affine: Optional[Tensor] = None) -> None:
        super().__init__(name=name, affine=None if affine is None else affine.detach().numpy())

    def _get_data(self, item: Any) -> ndarray:
        data: Tensor = item
        return super()._get_data(data.detach().cpu().numpy())


class TorchStorage(BaseStorage):
    """Save and load objects to disk using pytorch save and load"""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name}.pt")

    def save(self, item: Any, target_folder: str) -> None:
        save(item, self._get_path(target_folder))

    def load(self, target_folder: str, device: torch_device) -> Any:
        return load(self._get_path(target_folder), map_location=device)


class TensorCompressedStorage(BaseStorage):
    """Save and load pytorch Tensors using compressed numpy format"""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name}.npz")

    def save(self, item: Tensor, target_folder: str) -> None:
        numpy_savez(self._get_path(target_folder), array=item.detach().cpu().numpy())

    def load(self, target_folder: str, device: torch_device) -> Tensor:
        with numpy_load(self._get_path(target_folder)) as numpy_array:
            return as_tensor(numpy_array["array"], device=device)


class FloatStorage(BaseStorage):
    """Save and load float values"""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name}.npy")

    def save(self, item: float, target_folder: str) -> None:
        numpy_save(self._get_path(target_folder), item)

    def load(self, target_folder: str, device: torch_device) -> Any:
        return float(numpy_load(self._get_path(target_folder)))


class StringStorage(BaseStorage):
    """Save and load string values"""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name}.txt")

    def save(self, item: str, target_folder: str) -> None:
        with open(self._get_path(target_folder), mode="w", encoding="utf-8") as storage_file:
            storage_file.write(item)

    def load(self, target_folder: str, device: torch_device) -> Any:
        with open(self._get_path(target_folder), mode="r", encoding="utf-8") as storage_file:
            return storage_file.read()


class SequenceStorageWrapper(IStorage):
    """Wraps given storage into a storage that can save sequence of such items
    into separate subfolders"""

    def __init__(self, storage: IStorage, identifier: str, num_items: int) -> None:
        self._storage = storage
        self._identifier = identifier
        self._num_items = num_items

    def save(self, item: Sequence[Any], target_folder: str) -> None:
        for index, sequence_item in enumerate(item):
            indexed_target_folder = join(target_folder, self._get_subfolder_name_by_index(index))
            makedirs(indexed_target_folder, exist_ok=True)
            self._storage.save(sequence_item, indexed_target_folder)

    def load(self, target_folder: str, device: torch_device) -> list[Any]:
        items = []
        for index in range(self._num_items):
            items.append(
                self._storage.load(
                    join(target_folder, self._get_subfolder_name_by_index(index)), device
                )
            )
        return items

    def clear(self, target_folder: str) -> None:
        for index in range(self._num_items):
            indexed_target_folder = join(target_folder, self._get_subfolder_name_by_index(index))
            if self._storage.exists(indexed_target_folder):
                self._storage.clear(indexed_target_folder)
            if len(listdir(indexed_target_folder)) == 0:
                rmdir(indexed_target_folder)

    def exists(self, target_folder: str) -> bool:
        return all(
            self._storage.exists(join(target_folder, self._get_subfolder_name_by_index(index)))
            for index in range(self._num_items)
        )

    def _get_subfolder_name_by_index(self, index: int) -> str:
        return f"{self._identifier}_{str(index)}"


class OptionalStorageWrapper(IStorage):
    """Wraps given storage such that None value can be used"""

    def __init__(self, storage: IStorage, name: str) -> None:
        self._storage = storage
        self._name = name

    def _get_none_file_path(self, target_folder: str) -> str:
        return join(target_folder, f"{self._name}.none")

    def save(self, item: Any, target_folder: str) -> None:
        if item is None:
            open(self._get_none_file_path(target_folder), mode="a", encoding="utf-8").close()
        else:
            self._storage.save(item, target_folder)

    def load(self, target_folder: str, device: torch_device) -> Any:
        if isfile(self._get_none_file_path(target_folder)):
            return None
        return self._storage.load(target_folder, device)

    def clear(self, target_folder: str) -> None:
        none_file_path = self._get_none_file_path(target_folder)
        if isfile(self._get_none_file_path(target_folder)):
            remove(none_file_path)
        else:
            self._storage.clear(target_folder)

    def exists(self, target_folder: str) -> bool:
        return isfile(self._get_none_file_path(target_folder)) or self._storage.exists(
            target_folder
        )


def load_items_from_storages(
    target_folder: str,
    storages: Mapping[str, IStorage],
    device: torch_device,
    only_names: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Load data from dictionary of storages

    If only_names is given, only keys included in that set will be loaded.
    """
    items: dict[str, Any] = {}
    if only_names is None:
        names: Iterable[str] = storages.keys()
    else:
        names = only_names
    for name in names:
        if name in storages.keys():
            items[name] = storages[name].load(target_folder, device=device)
    return items


def save_items_to_storages(
    target_folder: str,
    storages: Mapping[str, IStorage],
    items: Mapping[str, Any],
    remove_not_completed_on_interrupt: bool = True,
) -> None:
    """Save data to dictionary of storages, keys of storages are assumed to be
    subset of keys of items"""
    for name in items.keys():
        try:
            storages[name].save(items[name], target_folder)
        except KeyboardInterrupt:
            if remove_not_completed_on_interrupt:
                storages[name].clear(target_folder)
            raise


def storages_exist(
    target_folder: str,
    storages: Iterable[IStorage],
) -> bool:
    """Return whether all the storages exist"""
    return all(storage.exists(target_folder) for storage in storages)
