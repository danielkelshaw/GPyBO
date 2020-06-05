import copy
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .kernel import Kernel
from .likelihood import GaussianLikelihood
from .utils.early_stopping import EarlyStopping
from .utils.shaping import to_tensor, uprank_two


class GP(nn.Module):

    def __init__(self, kernel: Kernel, train_noise: bool = False) -> None:

        super().__init__()

        self.kernel = kernel
        self.likelihood = GaussianLikelihood()

        self.x = None
        self.y = None

        self.noise = self._set_noise(train_noise)
        self.optimiser = torch.optim.Adam(self.kernel.parameters())

    def __call__(self, xp: Tensor) -> Tuple[Tensor, Tensor]:
        return self.predictive_posterior(xp)

    @staticmethod
    def _set_noise(noise: bool):
        if noise:
            return nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            return torch.tensor(0.0, dtype=torch.float32)

    def log_likelihood(self, stable: bool = True) -> Tensor:
        ll = self.likelihood.log_likelihood(
            self.kernel, self.x, self.y, self.noise, stable
        )
        return ll

    def optimise(self,
                 max_iter: int = 1000,
                 patience: int = 15,
                 ret_loss: bool = False) -> Optional[Tensor]:

        if self.x is None or self.y is None:
            raise ValueError('Must set (x, y) values first.')

        loss = None
        es = EarlyStopping(patience=patience)
        for i in range(max_iter):

            self.optimiser.zero_grad()

            loss = -self.log_likelihood(stable=False)

            loss.backward()
            self.optimiser.step()

            if es(loss):
                break

        if ret_loss:
            return loss

    def train(self, n_restarts: int = 10) -> None:

        best_loss = None
        best_params = {}

        for i in range(n_restarts):

            for param in self.parameters():
                torch.nn.init.uniform_(param, 0.0, 1.0)

            loss = self.optimise(ret_loss=True)

            if best_loss is None:
                best_loss = loss
            elif loss < best_loss:
                best_loss = loss
                p_dict = {k: v for k, v in self.named_parameters()}
                best_params = copy.deepcopy(p_dict)

        for name, param in self.named_parameters():
            param.data = best_params[name]

    @to_tensor
    @uprank_two
    def observe(self, x: Any, y: Any) -> None:

        if self.x is None and self.y is None:
            self.x = x
            self.y = y
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

        if not self.y.shape[1] == 1:
            raise ValueError('y must be a column vector')

    @to_tensor
    @uprank_two
    def predictive_posterior(self, xp: Any) -> Tuple[Tensor, Tensor]:

        k_xx = self.kernel.calculate(self.x, self.x)
        k_xxp = self.kernel.calculate(self.x, xp)
        k_xpx = self.kernel.calculate(xp, self.x)
        k_xpxp = self.kernel.calculate(xp, xp)

        k_xx_inv = torch.inverse(k_xx + self.noise * torch.eye(k_xx.shape[0]))

        p_mean = k_xpx @ k_xx_inv @ self.y
        p_covariance = k_xpxp - k_xpx @ k_xx_inv @ k_xxp

        return p_mean, p_covariance

    def __or__(self, other: Sequence[Any]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self
