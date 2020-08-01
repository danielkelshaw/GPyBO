from typing import NoReturn

import torch.nn as nn
from torch import Tensor


class BaseMean(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, xp: Tensor) -> Tensor:
        return self.calculate(xp)

    def __repr__(self) -> str:
        msg_list = [f'{k}={v:.3f}' for k, v in self.named_parameters()]
        msg = super().__repr__().replace('()', '(' + ', '.join(msg_list) + ')')
        return msg

    def __len__(self) -> int:
        return 1

    def calculate(self, xp: Tensor) -> NoReturn:

        """Produces the mean given a set of random variables.

        Parameters
        ----------
        xp : Tensor
            Set of random variables.

        Returns
        -------
        Tensor
            Calculated Mean.
        """

        raise NotImplementedError('BaseMean::calculate()')
