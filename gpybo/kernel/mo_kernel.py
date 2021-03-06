from typing import List

import torch
from torch import Tensor

from .base_kernel import BaseKernel


class MOKernel(BaseKernel):

    def __init__(self, kernels: List[BaseKernel]) -> None:

        """MultiOutput Kernel.

        Parameters
        ----------
        kernels : List[BaseKernel]
            List of kernels for multiple outputs.
        """

        super().__init__()

        for k in kernels:
            if not isinstance(k, BaseKernel):
                raise TypeError(f'{k} must be a Kernel.')

        self.kernels = kernels
        for idx, k in enumerate(self.kernels):
            self.add_module(str(idx), k)

    def __getitem__(self, item) -> BaseKernel:
        return self.kernels[item]

    def __len__(self) -> int:
        return len(self.kernels)

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:

        """Calculate multi-output kernel.

        Parameters
        ----------
        x : Tensor
            First set of random variables.
        xp : Tensor
            Second set of random variables.

        Returns
        -------
        output_kernel : Tensor
            Calculated multi-output kernel.
        """

        x_numel = len(self) * x.numel()
        xp_numel = len(self) * xp.numel()
        output_kernel = torch.zeros((x_numel, xp_numel), dtype=torch.float32)

        for idx, kernel in enumerate(self.kernels):

            x_start = idx * x.numel()
            x_end = x_start + x.numel()

            xp_start = idx * xp.numel()
            xp_end = xp_start + xp.numel()

            output_kernel[x_start : x_end, xp_start : xp_end] = kernel.calculate(x, xp)

        return output_kernel
