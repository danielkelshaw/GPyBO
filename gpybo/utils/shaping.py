import functools
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


def uprank_two(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    """Force Tensor inputs to fn to R2.

    Parameters
    ----------
    fn : Callable[[Any], Any]
        Function to decorate.

    Returns
    -------
    _decorated : Callable[[Any], Any]
        Decorated function.
    """

    @functools.wraps(fn)
    def _decorated(*args: Any) -> Any:
        return fn(*[uprank(x, 2) for x in args])

    return _decorated


def uprank(x: Any, n: int) -> Any:

    """Increase the rank of a Tensor, or simply return original object.

    Parameters
    ----------
    x : Any
        Object to increase the rank of.
    n : int
        Rank to achieve.

    Returns
    -------
    x : Any
        If x was a Tensor then it will now be of rank n.
        Otherwise, the original object will be returned.
    """

    if not isinstance(x, Tensor):
        return x

    if len(x.shape) > n:
        raise ValueError(f'rank of x already exceeds {n}...')

    while len(x.shape) < n:
        x = x.unsqueeze(dim=len(x.shape))

    return x


def to_tensor(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    """Force inputs to function to be Tensors.

    Parameters
    ----------
    fn : Callable[[Any], Any]
        Function to decorate.

    Returns
    -------
    _decorated : Callable[[Any], Any]
        Decorated function.
    """

    @functools.wraps(fn)
    def _decorated(*args: Any):
        return fn(*[convert_tensor(x) for x in args])

    return _decorated


def convert_tensor(x: Any) -> Any:

    """Convert the input to a Tensor.

    Parameters
    ----------
    x : Any
        Object to convert to Tensor.

    Returns
    -------
    x : Tensor
        Object converted to a Tensor.
    """

    if isinstance(x, Tensor):
        return x.type(torch.float32)
    elif isinstance(x, (int, float, list, tuple, np.ndarray)):
        return torch.tensor(x, dtype=torch.float32)
    else:
        return x


def to_array(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    """Force inputs to function to be np.ndarray.

    Parameters
    ----------
    fn : Callable[[Any], Any]
        Function to decorate.

    Returns
    -------
    _decorated : Callable[[Any], Any]
        Decorated function.
    """

    @functools.wraps(fn)
    def _decorated(*args: Any):
        return fn(*[convert_array(x) for x in args])

    return _decorated


def convert_array(x: Any) -> Any:

    """Convert the input to a np.ndarray.

    Parameters
    ----------
    x : Any
        Object to convert to np.ndarray.

    Returns
    -------
    x : np.ndarray
        Object converted to np.ndarray.
    """

    if isinstance(x, (int, float, list, tuple)):
        return np.array(x)
    elif isinstance(x, Tensor):
        return x.numpy()
    else:
        return x


def unwrap(x: Tensor) -> Tensor:

    """Unwrap Tensor to have a shape of (..., 1).

    Parameters
    ----------
    x : Tensor
        Tensor to unwrap.

    Returns
    -------
    Tensor
        Unwrapped Tensor.
    """

    return torch.stack(x.unbind(1)).view(-1, 1)
