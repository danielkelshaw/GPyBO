import torch
from torch import Tensor
from torch.distributions.normal import Normal

from .base_acquisition import BaseAcquisition
from ..gp import GP


# TODO :: Consider output shapes for acquisition functions.


class ExpectedImprovement(BaseAcquisition):

    def __init__(self, model: GP, alpha: float = 0.01) -> None:

        super().__init__(model)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:

        best_f = torch.max(self.model.y).to(x)
        posterior = self.model.posterior(x)

        mu = posterior.mean
        cov = posterior.covariance_matrix

        sigma = cov.diag().sqrt().clamp_min(1e-9).view(mu.shape)
        u = (mu - best_f.expand_as(mu) - self.alpha) / sigma

        norm = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = norm.cdf(u)
        updf = torch.exp(norm.log_prob(u))

        ei = sigma * (updf + u * ucdf)
        ei[torch.isnan(ei)] = 0.0

        return ei


class ProbabilityOfImprovement(BaseAcquisition):

    def __init__(self, model: GP) -> None:
        super().__init__(model)

    def forward(self, x: Tensor) -> Tensor:

        best_f = torch.max(self.model.y).to(x)
        posterior = self.model.posterior(x)

        mean = posterior.mean
        sigma = posterior.variance.sqrt()

        u = (mean - best_f.expand_as(mean)) / sigma
        norm = Normal(torch.zeros_like(u), torch.ones_like(u))

        return norm.cdf(u)
