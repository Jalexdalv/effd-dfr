from numpy import load, ndarray, save
from os import makedirs
from os.path import basename, exists
from torch import device, from_numpy, Tensor
from torch.nn import Module


def create_dir(path: str):
    if not exists(path=path):
        makedirs(name=path)


def get_device(module: Module) -> device:
    return next(module.parameters()).device


def convert_tensor_to_numpy(tensor: Tensor) -> ndarray:
    return tensor.detach().cpu().numpy()


def convert_numpy_to_tensor(array: ndarray) -> Tensor:
    return from_numpy(array).float()


def save_list(data: list, path: str) -> None:
    save(file=path, arr=data)


def load_list(path: str) -> list:
    return load(file=path, allow_pickle=True).tolist()


def get_file_name(path: str) -> str:
    return basename(p=path).split(sep='.')[0]
