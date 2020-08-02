from __future__ import annotations

import re
from typing import Any
from typing import NoReturn, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.shaping import convert_tensor


class BaseKernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        msg_list = [f'{k}={v:.3f}' for k, v in self.named_parameters()]
        msg = super().__repr__().replace('()', '(' + ', '.join(msg_list) + ')')
        return msg

    def __add__(self, other: Any) -> SumKernel:
        return SumKernel(self, other)

    def __radd__(self, other: Any) -> SumKernel:
        return SumKernel(other, self)

    def __sub__(self, other: Any) -> SumKernel:
        return SumKernel(self, -other)

    def __rsub__(self, other: Any) -> SumKernel:
        return SumKernel(other, -self)

    def __neg__(self) -> ProductKernel:
        return ProductKernel(-1, self)

    def __mul__(self, other: Any) -> ProductKernel:
        return ProductKernel(self, other)

    def __rmul__(self, other: Any) -> ProductKernel:
        return ProductKernel(other, self)

    def __len__(self) -> int:
        return 1

    def __call__(self, x: Tensor, xp: Tensor) -> Tensor:
        return self.calculate(x, xp)

    def calculate(self, x: Tensor, xp: Tensor) -> NoReturn:

        """Produces the Kernel given two sets of random variables.

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

        raise NotImplementedError('BaseKernel::calculate()')


class OneKernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.ones(x.shape[0], xp.shape[0], dtype=torch.float32)


class CombinationKernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def add(self, *args: BaseKernel) -> NoReturn:
        raise NotImplementedError('CombinationKernel::add()')

    def calculate(self, x: Tensor, xp: Tensor) -> NoReturn:
        raise NotImplementedError('CombinationKernel::calculate()')


class SumKernel(CombinationKernel):

    def __init__(self, *args: Union[BaseKernel, int, float, Tensor]) -> None:

        """Class for Summation of Multiple Kernels.

        Parameters
        ----------
        args : BaseKernel, int, float, Tensor
            Kernels to add.
        """

        super().__init__()

        self.kernels = []
        self.kernel_names = {}

        self.add(*args)

    def __repr__(self) -> str:
        msg = ' + '.join([str(k) for k in self.kernels])
        msg = re.sub(r'\+ -(\d*\.\d*)', r'- \1', msg)
        msg = re.sub(r'[^=]1\.0', ' ', msg)
        return msg

    def add(self, *args: Union[BaseKernel, int, float, Tensor]) -> None:

        """Adds Kernels to Class.

        Parameters
        ----------
        args : BaseKernel, int, float, Tensor
            Kernels to add to class
        """

        for idx, kernel in enumerate(args):

            if isinstance(kernel, SumKernel):
                self.__init__(*(self.kernels + kernel.kernels))

            elif isinstance(kernel, BaseKernel):
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

        """Calculates the output of the SumKernel.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        Tensor
            Calculated Kernel.
        """

        return torch.stack([kernel(x, xp) for kernel in self.kernels], dim=0).sum(dim=0)


class ProductKernel(CombinationKernel):

    def __init__(self, *args: Union[BaseKernel, int, float, Tensor]) -> None:

        """Class for Multiplication of Kernels.

        Parameters
        ----------
        args : BaseKernel, int, float, Tensor
            Values to use in kernel multiplication.
        """

        super().__init__()

        self.kernels = []
        self.kernel_names = {}

        self.add(*args)

    def __repr__(self) -> str:

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

    def add(self, *args: Union[BaseKernel, int, float, Tensor]) -> None:

        """Adds Kernels to Class.

        Parameters
        ----------
        args : BaseKernel, int, float, Tensor
            Kernels to add to class.
        """

        for idx, kernel in enumerate(args):

            if isinstance(kernel, ProductKernel):
                self.__init__(*(self.kernels + kernel.kernels))

            elif isinstance(kernel, BaseKernel):
                self.kernels.append(kernel)
                kname = str(kernel.__class__.__name__)
                self.kernel_names[kname] = self.kernel_names.get(kname, 0) + 1
                self.add_module(f'{kname}_{self.kernel_names[kname]}', kernel)

            elif isinstance(kernel, (int, float, Tensor)):
                self.kernels.append(convert_tensor(kernel))

            else:
                raise ValueError('Must add a Kernel')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        """Calculates the output of the ProductKernel.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        Tensor
            Calculated Kernel.
        """

        vals = []
        for kernel in self.kernels:
            if isinstance(kernel, BaseKernel):
                vals.append(kernel(x, xp))
            elif isinstance(kernel, Tensor):
                vals.append(kernel * torch.ones(x.numel(), xp.numel(), dtype=torch.float32))
            else:
                raise ValueError('Invalid Type.')

        return torch.stack(vals, dim=0).prod(dim=0)
