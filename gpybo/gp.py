from copy import deepcopy
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .distributions import Normal
from .kernel import Kernel, MOKernel
from .mean import Mean, ZeroMean, MOMean
from .likelihood import GaussianLikelihood
from .utils.early_stopping import EarlyStopping
from .utils.shaping import to_tensor, uprank_two, unwrap
from .utils.lab import pd_jitter


class GP(nn.Module):

    def __init__(self, kernel: Kernel, mean: Mean = ZeroMean(), train_noise: bool = False) -> None:

        """Gaussian Process Class.

        Parameters
        ----------
        kernel : Kernel
            Positive-Definite Kernel for calculations.
        train_noise : bool
            True if training with noise, False otherwise.
        """

        super().__init__()

        if not len(mean) == len(kernel):
            raise ValueError('Need same number of means and kernels.')

        self.mean = mean
        self.kernel = kernel
        self.likelihood = GaussianLikelihood()

        self.x = None
        self.y = None

        self.noise = self._set_noise(train_noise)
        self.optimiser = torch.optim.Adam

    def __repr__(self):
        return f'GP({str(self.mean)}, {str(self.kernel)})'

    def __or__(self, other: Sequence[Any]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self

    @to_tensor
    @uprank_two
    def __call__(self, xp: Tensor) -> Normal:
        return self.predictive_posterior(xp)

    @property
    def n_outputs(self) -> int:
        return len(self.mean)

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
            self.kernel, self.x, unwrap(self.y), self.noise, stable
        )
        return ll

    def optimise(self, max_iter: int = 1000, patience: int = 10) -> Tensor:

        """Optimise GP Parameters.

        Parameters
        ----------
        max_iter : int
            Number of iterations to run.
        patience : int
            Patience used in EarlyStopping.

        Returns
        -------
        loss : Tensor
            Calculated NLL.
        """

        if self.x is None or self.y is None:
            raise ValueError('Must set (x, y) values first.')

        loss = None
        optimiser = self.optimiser(self.parameters())
        es = EarlyStopping(patience=patience)

        for i in range(max_iter):

            loss = -self.log_likelihood(stable=False)

            if es(loss):
                break

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

        if not y.shape[1] == self.n_outputs:
            raise ValueError(
                f'shape of y ({y.shape}) does not match n_outputs.')

        if self.x is None and self.y is None:
            self.x = x
            self.y = y
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    @to_tensor
    @uprank_two
    def posterior(self, x: Any) -> MultivariateNormal:

        norm = self.predictive_posterior(x)
        mu, cov = norm.mu.flatten(), norm.covariance

        cov = pd_jitter(cov)
        mv_norm = MultivariateNormal(mu, cov)

        return mv_norm

    @to_tensor
    @uprank_two
    def predictive_posterior(self, xp: Any) -> Normal:

        """Predicts the Posterior.

        Parameters
        ----------
        xp : Tensor
            Features at which to predict the posterior.

        Returns
        -------
        Normal
            Normal distribution with calculated mu and covariance.
        """

        with torch.no_grad():

            k_xx = self.kernel.calculate(self.x, self.x)
            k_xxp = self.kernel.calculate(self.x, xp)
            k_xpx = self.kernel.calculate(xp, self.x)
            k_xpxp = self.kernel.calculate(xp, xp)

            k_xx_inv = torch.inverse(k_xx + self.noise * torch.eye(k_xx.shape[0]))

            p_mean = self.mean.calculate(xp) + k_xpx @ k_xx_inv @ unwrap(self.y)
            p_covariance = k_xpxp - k_xpx @ k_xx_inv @ k_xxp

        return Normal(mu=p_mean, covariance=p_covariance)
