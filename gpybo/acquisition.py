from typing import NoReturn

import torch
from torch import Tensor
from torch.distributions.normal import Normal

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

    def __init__(self, model: GP) -> None:
        super().__init__(model)

    @to_tensor
    @uprank_two
    def forward(self, x: Tensor) -> Tensor:

        best_f = torch.max(self.model.y).to(x)
        posterior_norm = self.model(x)
        posterior_mu = posterior_norm.mu
        posterior_cov = posterior_norm.covariance

        sigma = posterior_cov.diag().sqrt().clamp_min(1e-9).view(x.shape)
        u = (posterior_mu - best_f.expand_as(posterior_mu)) / sigma

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = sigma * (updf + u * ucdf)
        ei[torch.isnan(ei)] = 0.0

        return ei
