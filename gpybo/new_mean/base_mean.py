from torch import Tensor
import torch.nn as nn


class BaseMean(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, xp: Tensor) -> Tensor:
        raise NotImplementedError('BaseMean::calculate()')
