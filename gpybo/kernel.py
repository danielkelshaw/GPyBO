import torch
import torch.nn as nn
from torch import Tensor


class Kernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        raise NotImplementedError('Kernel::calculate()')

    def calculate_kernel(self, x: Tensor, xp: Tensor) -> Tensor:

        """Produces the Kernel given two sets of random variables.

        The calculation of the kernel depends on the kernel method being
        used - this is defined within the `self.calculate()` method. In
        order to calculate the kernel as efficiently as possible the RVs
        are converted to a pair of meshed grids - this ensures that the
        kernel calculation remains simple.

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

        return self.calculate(*torch.meshgrid([x, xp]))

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

        covariance[:x.numel(), :x.numel()] = self.calculate_kernel(x, x)
        covariance[:x.numel(), x.numel():] = self.calculate_kernel(x, xp)
        covariance[x.numel():, :x.numel()] = self.calculate_kernel(xp, x)
        covariance[x.numel():, x.numel():] = self.calculate_kernel(xp, xp)

        return covariance


class SquaredExponentialKernel(Kernel):

    def __init__(self) -> None:

        super().__init__()

        self.l = nn.Parameter(torch.tensor(1.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        """Squared Exponential Kernel Calculation.

        Function expects the two inputs: x, xp to have the same shape.
        This ensures that matrix operations can be carried out without
        any issues.

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

        if not x.shape == xp.shape:
            msg = f'x and xp must have the same shape - received: '
            msg += f'x with shape {x.shape} and '
            msg += f'xp with shape {xp.shape}'
            raise AssertionError(msg)

        return self.l ** 2 * torch.exp(-0.5 * torch.pow((x - xp) / self.sigma, 2))
