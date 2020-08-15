import torch
from torch import Tensor

from ..gp import GP
from .base_acquisition import BaseAcquisition


class qExpectedImprovement(BaseAcquisition):

    def __init__(self, model: GP, q: int = 1, alpha: float = 0.01) -> None:

        super().__init__(model)
        self.q = q
        self.alpha = alpha

    def forward(self, x: Tensor, nsamples: int = 512) -> Tensor:

        norm = self.model.posterior(x)
        samples = norm.rsample((self.q, nsamples))

        best_f = self.model.y.max()
        qei = (samples - best_f.expand_as(samples) - self.alpha).clamp_min(0.0)
        qei = qei.max(dim=1).values.mean(dim=0)

        return qei


class qProbabilityOfImprovement(BaseAcquisition):

    def __init__(self, model: GP, q: int = 1, alpha: float = 0.01) -> None:

        super().__init__(model)
        self.q = q
        self.alpha = alpha

    def forward(self, x: Tensor, nsamples: int = 512, temperature: float = 0.01) -> Tensor:

        norm = self.model.posterior(x)
        samples = norm.rsample((self.q, nsamples))

        best_f = self.model.y.max()
        best_sample = samples.max(dim=1).values

        poi = best_sample - best_f.expand_as(best_sample) - self.alpha
        poi = torch.sigmoid(poi / temperature).mean(dim=0)

        return poi


class qSimpleRegret(BaseAcquisition):

    def __init__(self, model: GP, q: int = 1) -> None:

        super().__init__(model)
        self.q = q

    def forward(self, x: Tensor, nsamples: int = 512) -> Tensor:

        norm = self.model.posterior(x)
        samples = norm.rsample((self.q, nsamples))

        sr = samples.max(dim=1).values.mean(dim=0)

        return sr
