import functools
from typing import Any, Callable

from torch import Tensor


def uprank(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:

    def _uprank(x: Any) -> Any:

        if not isinstance(x, Tensor):
            return x

        while len(x.shape) < 2:
            x = x.unsqueeze(dim=len(x.shape))
        return x

    @functools.wraps(fn)
    def _decorated(*args: Any) -> Any:
        return fn(*[_uprank(x) for x in args])

    return _decorated
