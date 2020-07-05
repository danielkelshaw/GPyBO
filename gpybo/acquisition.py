from typing import NoReturn

import torch
from torch import Tensor
from torch.distributions.normal import Normal

import numpy as np

from .gp import GP
from .utils.shaping import to_tensor, uprank_two


class BaseAcquisitionFunction:

    def __init__(self, model: GP) -> None:
        self.model = model

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> NoReturn:
        raise NotImplementedError('BaseAcquisitionFunction::forward()')


class ExpectedImprovement(BaseAcquisitionFunction):

    # TODO /> This needs reworking using a posterior, not N(0, 1)

    def __init__(self, model: GP, alpha: float = 0.01) -> None:
        super().__init__(model)
        self.alpha = alpha

    @to_tensor
    @uprank_two
    def forward(self, x: Tensor) -> Tensor:

        best_f = torch.max(self.model.y).to(x)
        posterior_norm = self.model(x)
        posterior_mu = posterior_norm.mu
        posterior_cov = posterior_norm.covariance

        sigma = posterior_cov.diag().sqrt().clamp_min(1e-9).view(x.shape)
        u = (posterior_mu - best_f.expand_as(posterior_mu) - self.alpha) / sigma

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = sigma * (updf + u * ucdf)
        ei[torch.isnan(ei)] = 0.0

        return ei


class ProbabilityOfImprovement(BaseAcquisitionFunction):

    def __init__(self, model: GP) -> None:
        super().__init__(model)

    @to_tensor
    @uprank_two
    def forward(self, x: Tensor) -> Tensor:

        if not x.numel() == 1:
            raise ValueError('PoI can only sample one value at a time.')

        mean = self.model.mean(x)
        cov = self.model.kernel(x, x)

        normal = Normal(mean, cov)

        return normal.cdf(x)


class qExpectedImprovement(BaseAcquisitionFunction):

    def __init__(self, model: GP, alpha: float = 0.01) -> None:
        super().__init__(model)
        self.alpha = alpha

    @to_tensor
    @uprank_two
    def forward(self, x: Tensor) -> Tensor:

        mv_norm = self.model.posterior(x)
        samples = mv_norm.rsample((1, 512)).requires_grad_(True)

        best_f = self.model.y.max()
        qei = (samples - best_f - self.alpha).clamp_min(0)
        qei = qei.max(dim=-2)[0].mean(dim=0)

        return qei


class qProbabilityOfImprovement(BaseAcquisitionFunction):

    def __init__(self, model: GP, alpha: float = 0.01) -> None:
        super().__init__(model)
        self.alpha = alpha

    @to_tensor
    @uprank_two
    def forward(self, x: Tensor) -> Tensor:

        temperature = 0.01
        mv_norm = self.model.posterior(x)
        samples = mv_norm.rsample((1, 512))

        best_f = self.model.y.max()
        max_sample = samples.max(dim=-2)[0]

        poi = max_sample - best_f - self.alpha
        poi = torch.sigmoid(poi / temperature).mean(dim=0)

        return poi


class qSimpleRegret(BaseAcquisitionFunction):

    def __init__(self, model: GP) -> None:
        super().__init__(model)

    def forward(self, x: Tensor) -> Tensor:

        mv_norm = self.model.posterior(x)
        samples = mv_norm.rsample(((1, 512)))

        sr = samples.max(dim=1)[0].mean(dim=0)

        return sr
