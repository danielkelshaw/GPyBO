from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .base_kernel import BaseKernel


class OneKernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.ones(x.shape[0], xp.shape[0], dtype=torch.float32)


class ZeroKernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], xp.shape[0], dtype=torch.float32)


class SquaredExponentialKernel(BaseKernel):

    def __init__(self, l: Any = 1.0, sigma: Any = 1.0) -> None:
        super().__init__()

        self.l = nn.Parameter(torch.tensor(l, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = torch.cdist(x, xp, p=2)
        return self.l ** 2 * torch.exp(-0.5 * torch.pow(dst / self.sigma, 2))


class SincKernel(BaseKernel):

    def __init__(self) -> None:
        super.__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = torch.cdist(x, xp, p=2)
        return torch.sin(dst) / dst


class Matern32Kernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = torch.cdist(x, xp, p=2)
        return (1 + dst) * torch.exp(-dst)


class Matern52Kernel(BaseKernel):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        dst = torch.cdist(x, xp, p=2)
        r1 = 5 ** 0.5 * dst
        r2 = 5 * dst ** 2 / 3

        return (1 + r1 + r2) * torch.exp(-r1)
