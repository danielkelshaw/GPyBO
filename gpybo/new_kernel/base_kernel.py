from typing import Any

import torch.nn as nn
from torch import Tensor

from .combination import SumKernel, ProductKernel


class BaseKernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
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

    def __neg__(self) -> 'ProductKernel':
        return ProductKernel(-1, self)

    def __mul__(self, other: Any) -> 'ProductKernel':
        return ProductKernel(self, other)

    def __rmul__(self, other: Any) -> 'ProductKernel':
        return ProductKernel(other, self)

    def __len__(self) -> int:
        return 1

    def __call__(self, x: Tensor, xp: Tensor) -> Tensor:
        return self.calculate(x, xp)

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        raise NotImplementedError('Kernel::calculate()')
