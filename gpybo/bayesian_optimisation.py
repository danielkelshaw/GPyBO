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
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

        self.acquisition = ExpectedImprovement
        self.opt_acquisition = torch.optim.Adam

        self.fn = fn

    def _optimise_acquisition(self) -> Tensor:

        _x = self.lb + (self.ub - self.lb) * torch.rand(self.lb.shape)
        _x.requires_grad_(True)

        ei_optimiser = self.opt_acquisition(params=[_x], lr=0.025)
        acquisition = self.acquisition(self.model)

        for i in range(1000):
            loss = -acquisition(_x)

            def closure():
                ei_optimiser.zero_grad()
                loss.backward()
                return loss

            ei_optimiser.step(closure)

        return _x.requires_grad_(False)

    def optimise(self) -> NoReturn:
        raise NotImplementedError('BO::optimise()')
