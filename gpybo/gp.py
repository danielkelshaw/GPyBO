from copy import deepcopy
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .kernel.base_kernel import BaseKernel
from .mean.base_mean import BaseMean
from .mean.means import ZeroMean

from .utils.lab import pd_jitter
from .utils.shaping import uprank_two, to_tensor

"""
At the moment the implementation of `GP` is rather messy and the inputs
are not consistent - this is leading to troubles with implementation of
Bayesian Optimisation.

This stage of work aims to focus on improving the quality of the GP
code such that it is not a blocker for the BO project.


// Stage One:
- [x] Get everything working with 'hardcoded' shapes.
- [x] Implement additional tests to make sure these work as intended.

// Stage Two:
- [x] Add decorators to allow less constrained input (for users).
- [x] Implement tests for decorators to ensure they work as intended.
- [x] Add user-friendly interfaces such as __call__ and __repr__.
"""


class GP(nn.Module):

    def __init__(self,
                 kernel: BaseKernel,
                 mean: BaseMean = ZeroMean(),
                 noise: bool = False) -> None:

        super().__init__()

        if not len(mean) == len(kernel):
            raise ValueError('len(mean) must equal len(kernel).')

        self.mean = mean
        self.kernel = kernel

        self.x = None
        self.y = None

        self.noise = self._set_noise(noise)
        self.optimiser = torch.optim.Adam

    def __repr__(self) -> str:
        return f'GP({str(self.mean)}, {str(self.kernel)})'

    def __or__(self, other: Tuple[Tensor, Tensor]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self

    def __call__(self, xp: Any) -> MultivariateNormal:
        return self.posterior(xp)

    @staticmethod
    def _set_noise(noise: bool) -> Tensor:

        if noise:
            return nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            return torch.tensor(0.0, dtype=torch.float32)

    @to_tensor
    @uprank_two
    def observe(self, x: Any, y: Any) -> None:

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Must provide 2D inputs.')
        if not x.shape[0] == y.shape[0]:
            raise ValueError('Must provide same number of samples for x and y.')

        if self.x is None and self.y is None:
            self.x = x
            self.y = y
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    @to_tensor
    @uprank_two
    def posterior(self, xp: Any) -> MultivariateNormal:

        if not xp.shape[1] == self.x.shape[1]:
            raise ValueError('xp shape does not match observed sample shapes.')

        k_xx = self.kernel.calculate(self.x, self.x)
        k_xxp = self.kernel.calculate(self.x, xp)
        k_xpx = self.kernel.calculate(xp, self.x)
        k_xpxp = self.kernel.calculate(xp, xp)

        k_xx_inv = torch.inverse(k_xx + self.noise * torch.eye(k_xx.shape[0]))

        p_mean = self.mean.calculate(xp) + k_xpx @ k_xx_inv @ self.y
        p_covariance = k_xpxp - k_xpx @ k_xx_inv @ k_xxp

        p_mean = p_mean.flatten()
        p_covariance = pd_jitter(p_covariance)

        return MultivariateNormal(p_mean, p_covariance)

    def optimise(self, n_iterations: int = 1000) -> Tensor:

        loss = None
        optimiser = self.optimiser(self.parameters())

        for i in range(n_iterations):

            optimiser.zero_grad()

            loss = -self.log_likelihood(grad=True).sum()

            loss.backward()
            optimiser.step()

        return loss

    def optimise_restarts(self, n_restarts: int = 10, n_iterations: int = 1000) -> Tensor:

        losses = []
        params = []

        for i in range(n_restarts):

            for param in self.parameters():
                torch.nn.init.uniform_(param, 0.0, 2.0)

            loss = self.optimise(n_iterations=n_iterations)

            losses.append(loss)
            params.append(deepcopy({k: v for k, v in self.named_parameters()}))

        losses = torch.stack(losses, dim=0)
        idx_best = losses.argmin()

        for name, param in self.named_parameters():
            param.data = params[idx_best][name]

        return losses[idx_best]

    def log_likelihood(self, grad: bool = False) -> Tensor:

        k_xx = self.kernel.calculate(self.x, self.x)
        k_xx = k_xx + self.noise * torch.eye(k_xx.shape[0])
        k_xx = pd_jitter(k_xx)

        if grad:
            log_term = 0.5 * torch.log(1e-6 + torch.det(k_xx))
            y_term = 0.5 * self.y.T @ torch.inverse(k_xx) @ self.y
            const_term = 0.5 * len(self.x) * np.log(2 * np.pi)

            ll = -y_term - log_term - const_term

        else:
            L = torch.cholesky(k_xx)
            a0, _ = torch.lstsq(self.y, L)
            alpha, _ = torch.lstsq(a0, L.T)

            y_alpha = -0.5 * self.y * alpha.view_as(self.y)
            trace_log = torch.trace(torch.log(L))
            const = 0.5 * len(self.x) * np.log(2 * np.pi)

            ll = y_alpha - trace_log - const

        return ll
