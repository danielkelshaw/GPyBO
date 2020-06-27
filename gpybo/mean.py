import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .utils.shaping import to_tensor, uprank_two


class Mean(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, xp: Tensor) -> Tensor:
        return self.calculate(xp)

    def __repr__(self):
        msg_list = [f'{k}={v:.3f}' for k, v in self.named_parameters()]
        msg = super().__repr__().replace('()', '(' + ', '.join(msg_list) + ')')
        return msg

    def calculate(self, xp: Tensor) -> Tensor:
        raise NotImplementedError('Mean::calculate()')


class ZeroMean(Mean):

    def __init__(self) -> None:
        super().__init__()

    @to_tensor
    @uprank_two
    def calculate(self, xp: Tensor) -> Tensor:
        return torch.zeros_like(xp)


class MOMean(Mean):

    def __init__(self, means: List[Mean]) -> None:

        super().__init__()
        self.means = means
        for idx, m in enumerate(self.means):
            self.add_module(str(idx), m)

    @property
    def n_means(self):
        return len(self.means)

    def calculate(self, xp: Tensor) -> Tensor:

        output_mean = torch.zeros((self.n_means * xp.numel(), 1), dtype=torch.float32)

        for idx, mean in enumerate(self.means):

            xp_start = idx * xp.numel()
            xp_end = xp_start + xp.numel()

            output_mean[xp_start : xp_end] = mean.calculate(xp)

        return output_mean
