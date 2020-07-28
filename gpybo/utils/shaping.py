import functools
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


def uprank_two(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    @functools.wraps(fn)
    def _decorated(*args: Any) -> Any:
        return fn(*[uprank(x, 2) for x in args])

    return _decorated


def uprank(x: Any, n: int) -> Any:

    if not isinstance(x, Tensor):
        return x

    if len(x.shape) > n:
        raise ValueError(f'rank of x already exceeds {n}...')

    while len(x.shape) < n:
        x = x.unsqueeze(dim=len(x.shape))

    return x


def to_tensor(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    @functools.wraps(fn)
    def _decorated(*args: Any):
        return fn(*[convert_tensor(x) for x in args])

    return _decorated


def convert_tensor(x: Any) -> Any:

    if isinstance(x, Tensor):
        return x.type(torch.float32)
    elif isinstance(x, (int, float, list, tuple, np.ndarray)):
        return torch.tensor(x, dtype=torch.float32)
    else:
        return x


def to_array(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    @functools.wraps(fn)
    def _decorated(*args: Any):
        return fn(*[convert_array(x) for x in args])

    return _decorated()


def convert_array(x: Any) -> Any:

    if isinstance(x, (int, float, list, tuple)):
        return np.array(x)
    elif isinstance(x, Tensor):
        return x.numpy()
    else:
        return x


def unwrap(x: Tensor):
    return torch.stack(x.unbind(1)).view(-1, 1)
