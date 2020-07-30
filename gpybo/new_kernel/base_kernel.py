import torch
import torch.nn as nn
from torch import Tensor
from typing import Any


class BaseKernel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, x: Tensor, xp: Tensor) -> Tensor:
        raise NotImplementedError('Kernel::calculate()')
