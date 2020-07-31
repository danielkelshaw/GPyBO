import re
from typing import NoReturn, Union

import torch
from torch import Tensor

from .base_kernel import BaseKernel
from .kernel import OneKernel
from ..utils.shaping import convert_tensor


class CombinationKernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def add(self, *args: BaseKernel) -> NoReturn:
        raise NotImplementedError('CombinationKernel::add()')

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
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

    def __repr__(self):
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
