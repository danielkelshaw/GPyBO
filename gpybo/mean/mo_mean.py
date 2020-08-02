from typing import List

import torch
from torch import Tensor

from .base_mean import BaseMean


class MOMean(BaseMean):

    def __init__(self, means: List[BaseMean]) -> None:

        """Multi-output Mean.

        Parameters
        ----------
        means : List[BaseMean]
            List of means to use for multiple outputs.
        """

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

        """Calculate the multi-output mean.

        Parameters
        ----------
        xp : Tensor
            Set of random variables.

        Returns
        -------
        output_mean : Tensor
            Calculated multi-output mean.
        """

        output_mean = torch.zeros((len(self) * xp.numel(), 1), dtype=torch.float32)

        for idx, mean in enumerate(self.means):

            xp_start = idx * xp.numel()
            xp_end = xp_start + xp.numel()

            output_mean[xp_start: xp_end] = mean.calculate(xp)

        return output_mean
