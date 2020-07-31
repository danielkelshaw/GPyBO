from typing import List

import torch
from torch import Tensor

from .base_mean import BaseMean


class MOMean(BaseMean):

    def __init__(self, means: List[BaseMean]) -> None:

        super().__init__()

        for m in means:
            if not isinstance(m, BaseMean):
                raise TypeError(f'{m} must be a Mean.')

        self.means = means
        for idx, m in enumerate(self.means):
            self.add_module(str(idx), m)

    def __getitem__(self, item) -> BaseMean:
        return self.means[item]

    def __len__(self) -> int:
        return len(self.means)

    def calculate(self, xp: Tensor) -> Tensor:

        output_mean = torch.zeros((self.n_means * xp.numel(), 1), dtype=torch.float32)

        for idx, mean in enumerate(self.means):

            xp_start = idx * xp.numel()
            xp_end = xp_start + xp.numel()

            output_mean[xp_start: xp_end] = mean.calculate(xp)

        return output_mean
