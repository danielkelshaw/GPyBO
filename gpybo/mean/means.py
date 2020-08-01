import torch
from torch import Tensor

from .base_mean import BaseMean


class ZeroMean(BaseMean):

    def __init__(self) -> None:

        """Zero Mean."""

        super().__init__()

    def calculate(self, xp: Tensor) -> Tensor:
        return torch.zeros(xp.shape[0], 1)
