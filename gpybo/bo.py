from typing import Tuple

import torch
from torch import Tensor

from .gp import GP
from .acquisition.analytic import ExpectedImprovement
from .utils.samplers import draw_sobol
from .utils.optim import scipy_acqopt


class NewBO:

    def __init__(self, model: GP, lb: Tensor, ub: Tensor) -> None:

        self.model = model

        self.lb = lb
        self.ub = ub

        self.acquisition = ExpectedImprovement

    def optimise_acquisition(self, nsamples: int = 512) -> Tuple[Tensor, Tensor]:

        acq = self.acquisition(self.model)
        x_draw = draw_sobol(self.lb, self.ub, n=nsamples)
        acq_draw = acq(x_draw)

        xopt = x_draw[acq_draw.argmax()].unsqueeze(0)
        xnew, loss = scipy_acqopt(xopt, self.lb, self.ub, acq)

        return xnew.unsqueeze(0), loss

    def optimise_acquisition_restarts(self, n_restarts: int = 10, nsamples: int = 512) -> Tuple[Tensor, Tensor]:

        losses = []
        candidates = []

        for i in range(n_restarts):

            _x, _loss = self.optimise_acquisition(nsamples)

            losses.append(_loss)
            candidates.append(_x)

        losses = torch.cat(losses)
        candidates = torch.cat(candidates)

        best = losses.argmin()

        return candidates[best].unsqueeze(0), losses[best]
