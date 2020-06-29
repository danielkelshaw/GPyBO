import numpy as np
import torch
from torch import Tensor

from .kernel import Kernel


class Likelihood:

    def __init__(self) -> None:
        pass

    def log_likelihood(self, *args):
        raise NotImplementedError('Likelihood::log_likelihood()')


class GaussianLikelihood(Likelihood):

    def __init__(self) -> None:
        super().__init__()

    def log_likelihood(self, kernel: Kernel, x: Tensor, y: Tensor, noise: Tensor, stable: bool = True) -> Tensor:

        """Calculates the Log Likelihood for the Gaussian Prior.

        Parameters
        ----------
        kernel : Kernel
            Kernel used to calculate covariance.
        x : Tensor
            Input values.
        y : Tensor
            Corresponding output values.
        noise : Tensor
            Noise in the GP.
        stable : bool
            True if stable calculation, False otherwise.

        Returns
        -------
        ll : Tensor
            Log Likelihood of outputs given the data.
        """

        kern = kernel.calculate(x, x)
        K = kern + noise * torch.eye(kern.shape[0])

        if stable:
            L = torch.cholesky(K)

            a0, _ = torch.lstsq(y, L)
            alpha, _ = torch.lstsq(a0, L.T)

            y_alpha = -0.5 * y * alpha.view_as(y)
            trace_log = torch.trace(torch.log(L))
            const = 0.5 * len(x) * np.log(2 * np.pi)

            ll = y_alpha - trace_log - const
        else:
            log_term = 0.5 * torch.log(torch.det(K))
            y_term = 0.5 * y.T @ torch.inverse(K) @ y
            const_term = 0.5 * len(x) * np.log(2 * np.pi)

            ll = -y_term - log_term - const_term

        return ll.flatten()
