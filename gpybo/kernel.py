import torch
import torch.nn as nn
from torch import Tensor

import re
from typing import Any, List, Union

from .utils.lab import pw_dist
from .utils.shaping import convert_tensor


class Kernel(nn.Module):

    def __init__(self, *args: Union[int, float]) -> None:
        super().__init__()

    def __repr__(self):
        msg_list = [f'{k}={v:.3f}' for k, v in self.named_parameters()]
        msg = super().__repr__().replace('()', '(' + ', '.join(msg_list) + ')')
        return msg

    def __add__(self, other: Any) -> 'SumKernel':
        return SumKernel(self, other)

    def __radd__(self, other: Any) -> 'SumKernel':
        return SumKernel(other, self)

    def __sub__(self, other: Any) -> 'SumKernel':
        return SumKernel(self, -other)

    def __rsub__(self, other: Any) -> 'SumKernel':
        return SumKernel(other, -self)

    def __neg__(self):
        return ProductKernel(-1, self)

    def __mul__(self, other: Any) -> 'ProductKernel':
        return ProductKernel(self, other)

    def __rmul__(self, other: Any) -> 'ProductKernel':
        return ProductKernel(other, self)

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

    def __init__(self,
                 l: Union[int, float] = 1.0,
                 sigma: Union[int, float] = 1.0) -> None:

        """Squared Exponential Kernel

        Parameters
        ----------
        l : int, float, Tensor
            Lengthscale for Kernel
        sigma : int, float, Tensor
            Noise for Kernel.
        """

        super().__init__()

        self.l = nn.Parameter(convert_tensor(l))
        self.sigma = nn.Parameter(convert_tensor(sigma))

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


class CombinationKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

    def add(self, *args: Kernel):
        raise NotImplementedError('CombinationKernel::add()')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        raise NotImplementedError('CombinationKernel::calculate()')


class SumKernel(CombinationKernel):

    def __init__(self, *args: Kernel) -> None:

        """Class for Summation of Multiple Kernels.

        Parameters
        ----------
        args : Kernel
            Kernels to add.
        """

        super().__init__()

        self.kernels: List[Kernel] = []
        self.kernel_names = {}

        self.add(*args)

    def __repr__(self):
        msg = ' + '.join([str(k) for k in self.kernels])
        msg = re.sub(r'\+ -(\d*\.\d*)', r'- \1', msg)
        msg = re.sub(r'[^=]1\.0', ' ', msg)
        return msg


    def add(self, *args: Kernel) -> None:

        """Adds Kernels to Class.

        Parameters
        ----------
        args : Kernel
            Kernels to add to class
        """

        for idx, kernel in enumerate(args):

            if isinstance(kernel, SumKernel):
                self.__init__(*(self.kernels + kernel.kernels))

            elif isinstance(kernel, Kernel):
                self.kernels.append(kernel)
                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1
                self.add_module(f'{kname}_{self.kernel_names[kname]}', kernel)

            elif isinstance(kernel, (int, float, Tensor)):
                tmp_kernel = kernel * OneKernel()
                self.kernels.append(tmp_kernel)
                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1

            else:
                raise ValueError('Must add a Kernel')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.stack([kernel(x, xp) for kernel in self.kernels], dim=0).sum(dim=0)


class ProductKernel(CombinationKernel):

    def __init__(self, *args: Union[Kernel, int, float, Tensor]) -> None:

        """Class for Multiplication of Kernels.

        Parameters
        ----------
        args : Kernel, int, float, Tensor
            Values to use in kernel multiplication.
        """

        super().__init__()

        self.kernels: List[Union[Kernel, Tensor]] = []
        self.kernel_names = {}

        self.add(*args)

    def __repr__(self):

        consts = [1.0]
        msg_list = []
        for k in self.kernels:
            if not isinstance(k, CombinationKernel):
                if isinstance(k, Tensor):
                    consts[0] *= float(k)
                else:
                    msg_list.append(str(k))
            else:
                msg_list.append('[' + str(k) + ']')

        consts = list(map(str, consts + msg_list))
        msg = ''.join(consts)

        msg = re.sub(r'^1.0', '', msg)
        return msg

    def add(self, *args: Union[Kernel, int, float, Tensor]) -> None:

        """Adds Kernels to Class.

        Parameters
        ----------
        args : Kernel
            Kernels to add to class.
        """

        for idx, kernel in enumerate(args):

            if isinstance(kernel, ProductKernel):
                self.__init__(*(self.kernels + kernel.kernels))

            elif isinstance(kernel, Kernel):
                self.kernels.append(kernel)
                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1
                self.add_module(f'{kname}_{self.kernel_names[kname]}', kernel)

            elif isinstance(kernel, (int, float, Tensor)):
                self.kernels.append(convert_tensor(kernel))

            else:
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


class MOKernel(Kernel):

    def __init__(self, kernels: List[Kernel]) -> None:

        super().__init__()

        for k in kernels:
            if not isinstance(k, Kernel):
                raise TypeError(f'{k} must be a Kernel.')

        self.kernels = kernels
        for idx, k in enumerate(self.kernels):
            self.add_module(str(idx), k)

    def __getitem__(self, item):
        return self.kernels[item]

    def __len__(self):
        return len(self.kernels)

    @property
    def n_kernels(self) -> int:
        return len(self.kernels)

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        x_numel = self.n_kernels * x.numel()
        xp_numel = self.n_kernels * xp.numel()
        output_kernel = torch.zeros((x_numel, xp_numel), dtype=torch.float32)

        for idx, kernel in enumerate(self.kernels):

            x_start = idx * x.numel()
            x_end = x_start + x.numel()

            xp_start = idx * xp.numel()
            xp_end = xp_start + xp.numel()

            output_kernel[x_start : x_end, xp_start : xp_end] = kernel.calculate(x, xp)

        return output_kernel
