from typing import Any, NoReturn, Sequence, Tuple

import torch
from torch import Tensor

from .kernel import Kernel
from .likelihood import GaussianLikelihood
from .utils.early_stopping import EarlyStopping


class GP:

    def __init__(self, kernel: Kernel) -> None:

        self.kernel = kernel
        self.likelihood = GaussianLikelihood()

        self.x = None
        self.y = None

        self.optimiser = torch.optim.Adam(self.kernel.parameters())

    def log_likelihood(self, stable) -> Tensor:
        ll = self.likelihood.log_likelihood(self.kernel, self.x, self.y, stable)
        return ll

    def train(self) -> NoReturn:
        raise NotImplementedError('GP::train()')

    def optimise(self, max_iterations: int = 2000, patience: int = 15) -> None:

        if self.x is None or self.y is None:
            raise ValueError('Must set (x, y) values first.')

        es = EarlyStopping(patience=patience)
        for i in range(max_iterations):

            self.optimiser.zero_grad()

            loss = -self.log_likelihood(stable=False)

            loss.backward()
            self.optimiser.step()

            if es(loss):
                break

    def observe(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y.unsqueeze(1)

    def predictive_posterior(self, xp: Tensor) -> Tuple[Tensor, Tensor]:

        k_xx = self.kernel.calculate_kernel(self.x, self.x)
        k_xxp = self.kernel.calculate_kernel(self.x, xp)
        k_xpx = self.kernel.calculate_kernel(xp, self.x)
        k_xpxp = self.kernel.calculate_kernel(xp, xp)

        k_xx_inv = torch.inverse(k_xx)

        p_mean = k_xpx @ k_xx_inv @ self.y
        p_covariance = k_xpxp - k_xpx @ k_xx_inv @ k_xxp

        return p_mean, p_covariance

    def __or__(self, other: Sequence[Any]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self
