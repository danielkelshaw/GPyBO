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

    def _decorated(*args: Any):
        return fn(*[_tensor(x) for x in args])

    return _decorated


def _tensor(x: Any) -> Any:

    if isinstance(x, Tensor):
        return x.type(torch.float32)
    elif isinstance(x, (int, float, list, tuple, np.ndarray)):
        return torch.tensor(x, dtype=torch.float32)
    else:
        return x
