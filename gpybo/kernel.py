import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Union

from .utils.lab import pw_dist


class Kernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, xp: Tensor) -> Tensor:
        return self.calculate(x, xp)

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


class OneKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.ones(x.numel(), xp.numel(), dtype=torch.float32)


class ZeroKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.zeros(x, xp, dtype=torch.float32)


class SincKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = pw_dist(x, xp)
        return torch.sin(dst) / dst


class Matern32Kernel(Kernel):

    def __init__(self):
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = pw_dist(x, xp)
        return (1 + dst) * torch.exp(-dst)


class Matern52Kernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = pw_dist(x, xp)
        r1 = 5 ** 0.5 * dst
        r2 = 5 * dst ** 2 / 3

        return (1 + r1 + r2) * torch.exp(-r1)


class SumKernel(Kernel):

    def __init__(self, *args: Kernel) -> None:
        super().__init__()

        self.kernels: List[Kernel] = []
        self.kernel_names = {}

        self.add(*args)

    def add(self, *args: Kernel) -> None:
        for idx, kernel in enumerate(args):
            if isinstance(kernel, Kernel):
                self.kernels.append(kernel)

                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1

                self.add_module(f'{kname}_{self.kernel_names[kname]}', kernel)
            else:
                raise ValueError('Must add a Kernel')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.stack([kernel(x, xp) for kernel in self.kernels], dim=0).sum(dim=0)


class ProductKernel(Kernel):

    def __init__(self, *args: Union[Kernel, float]) -> None:
        super().__init__()

        self.kernels: List[Union[Kernel, Tensor]] = []
        self.kernel_names = {}

        self.add(*args)

    def add(self, *args: Union[Kernel, float]) -> None:
        for idx, kernel in enumerate(args):
            if isinstance(kernel, Kernel):
                self.kernels.append(kernel)

                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1

                self.add_module(f'{kname}_{self.kernel_names[kname]}', kernel)
            elif isinstance(kernel, float):
                self.kernels.append(torch.tensor(kernel, dtype=torch.float32))
            else:
                print(kernel)
                print(type(kernel))
                raise ValueError('Must add a Kernel')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        vals = []
        for kernel in self.kernels:
            if isinstance(kernel, Kernel):
                vals.append(kernel(x, xp))
            elif isinstance(kernel, Tensor):
                vals.append(kernel * torch.ones(x.numel(), xp.numel(), dtype=torch.float32))
            else:
                raise ValueError('Invalid Type.')

        return torch.stack(vals, dim=0).prod(dim=0)
