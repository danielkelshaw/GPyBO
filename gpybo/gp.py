from copy import deepcopy
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .kernel import Kernel
from .mean import Mean, ZeroMean
from .likelihood import GaussianLikelihood
from .utils.shaping import to_tensor, uprank_two


class GP(nn.Module):

    def __init__(self, kernel: Kernel, train_noise: bool = False) -> None:

        """Gaussian Process Class.

        Parameters
        ----------
        kernel : Kernel
            Positive-Definite Kernel for calculations.
        train_noise : bool
            True if training with noise, False otherwise.
        """

        super().__init__()

        self.mean = ZeroMean()
        self.kernel = kernel
        self.likelihood = GaussianLikelihood()

        self.x = None
        self.y = None

        self.noise = self._set_noise(train_noise)
        self.optimiser = torch.optim.Adam

    def __repr__(self):
        return f'GP({str(self.mean)}, {str(self.kernel)})'

    @to_tensor
    @uprank_two
    def __call__(self, xp: Tensor) -> Tuple[Tensor, Tensor]:
        return self.predictive_posterior(xp)

    @staticmethod
    def _set_noise(noise: bool) -> Tensor:

        """Sets Noise to either Zero / a Parameter.

        Parameters
        ----------
        noise : bool
            True if noise is to be a parameter, False otherwise.

        Returns
        -------
        Tensor
            Parameter if noise is required, Zero otherwise.
        """

        if noise:
            return nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            return torch.tensor(0.0, dtype=torch.float32)

    def log_likelihood(self, stable: bool = True) -> Tensor:

        """Calculates the Log Likelihood of the Observations.

        Parameters
        ----------
        stable : bool
            True if using autograd, False otherwise.

        Returns
        -------
        ll : Tensor
            Log Likelihood of the observed events.
        """

        ll = self.likelihood.log_likelihood(
            self.kernel, self.x, self.y, self.noise, stable
        )
        return ll

    def optimise(self, max_iter: int = 1000) -> Tensor:

        """Optimise GP Parameters.

        Parameters
        ----------
        max_iter : int
            Number of iterations to run.

        Returns
        -------
        loss : Tensor
            Calculated NLL.
        """

        if self.x is None or self.y is None:
            raise ValueError('Must set (x, y) values first.')

        loss = None
        optimiser = self.optimiser(self.parameters())
        for i in range(max_iter):

            loss = -self.log_likelihood(stable=False)

            def closure():
                optimiser.zero_grad()
                loss.backward()
                return loss

            optimiser.step(closure)

        return loss

    def train(self, n_restarts: int = 10, n_iterations: int = 2000) -> Tensor:

        """Train the GP Model.

        Parameters
        ----------
        n_restarts : int
            Number of times to restart the optimisation process.
        n_iterations : int
            Number of iterations to optimise for.

        Returns
        -------
        loss : Tensor
            Calculated NLL of the best optimisation attempt.
        """

        best_loss = None
        best_params = deepcopy({k: v for k, v in self.named_parameters()})

        for i in range(n_restarts):

            for param in self.parameters():
                torch.nn.init.uniform_(param, 0.0, 1.0)

            loss = self.optimise(max_iter=n_iterations)

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_params = deepcopy({k: v for k, v in self.named_parameters()})

        for name, param in self.named_parameters():
            param.data = best_params[name]

        return best_loss

    @to_tensor
    @uprank_two
    def observe(self, x: Any, y: Any) -> None:

        """Adds additional observations.

        Parameters
        ----------
        x : Tensor
            Observation features.
        y : Tensor
            Observations.
        """

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

        """Predicts the Posterior.

        Parameters
        ----------
        xp : Tensor
            Features at which to predict the posterior.

        Returns
        -------
        p_mean : Tensor
            Mean of the predictive posterior.
        p_cov : Tensor
            Covariance of the predictive posterior.
        """

        k_xx = self.kernel.calculate(self.x, self.x)
        k_xxp = self.kernel.calculate(self.x, xp)
        k_xpx = self.kernel.calculate(xp, self.x)
        k_xpxp = self.kernel.calculate(xp, xp)

        k_xx_inv = torch.inverse(k_xx + self.noise * torch.eye(k_xx.shape[0]))

        p_mean = self.mean.calculate(xp) + k_xpx @ k_xx_inv @ self.y
        p_covariance = k_xpxp - k_xpx @ k_xx_inv @ k_xxp

        return p_mean, p_covariance

    def __or__(self, other: Sequence[Any]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self
