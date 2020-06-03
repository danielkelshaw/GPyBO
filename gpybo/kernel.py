import torch
import torch.nn as nn
from torch import Tensor

from .utils.lab import pw_dist


class Kernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        """Produces the Kernel given two sets of random variables.

        The calculation of the kernel depends on the kernel method being
        used - this is defined within the defined super class.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        Tensor
            Calculated kernel.
        """

        raise NotImplementedError('Kernel::calculate()')

    def covariance(self, x: Tensor, xp: Tensor) -> Tensor:

        """Calculates the Covariance Matrix.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        covariance : Tensor
            Calculated covariance matrix.
        """

        if not x.ndim == xp.ndim == 1:
            raise AssertionError('x and xp must be one-dimensional.')

        n = x.numel() + xp.numel()
        covariance = torch.zeros((n, n))

        covariance[:x.numel(), :x.numel()] = self.calculate(x, x)
        covariance[:x.numel(), x.numel():] = self.calculate(x, xp)
        covariance[x.numel():, :x.numel()] = self.calculate(xp, x)
        covariance[x.numel():, x.numel():] = self.calculate(xp, xp)

        return covariance


class SquaredExponentialKernel(Kernel):

    def __init__(self) -> None:

        super().__init__()

        self.l = nn.Parameter(torch.tensor(1.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        """Squared Exponential Kernel Calculation.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        Tensor
            Calculated kernel.
        """

        dst = pw_dist(x, xp)
        return self.l ** 2 * torch.exp(-0.5 * torch.pow(dst / self.sigma, 2))
