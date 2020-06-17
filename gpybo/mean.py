import torch
from torch import Tensor

from .utils.shaping import to_tensor, uprank_two


class Mean:

    def __init__(self) -> None:
        pass

    def calculate(self, xp: Tensor) -> Tensor:
        raise NotImplementedError('Mean::calculate()')


class ZeroMean(Mean):

    def __init__(self) -> None:
        super().__init__()

    @to_tensor
    @uprank_two
    def calculate(self, xp: Tensor) -> Tensor:
        return torch.zeros_like(xp)
