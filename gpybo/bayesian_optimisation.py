from typing import Any, Callable, NoReturn

import torch
from torch import Tensor

from .gp import GP
from .acquisition import ExpectedImprovement


class BO:

    def __init__(self,
                 model: GP,
                 bounds: Tensor,
                 fn: Callable[[Any], Any]) -> None:

        self.model = model
        self.bounds = bounds
        self.acquisition = ExpectedImprovement(self.model)
        self.opt_acquisition = torch.optim.Adam

        self.fn = fn

    def _optimise_acquisition(self) -> NoReturn:
        raise NotImplementedError('BO::_optimise_acquisition()')

    def optimise(self) -> NoReturn:
        raise NotImplementedError('BO::optimise()')
