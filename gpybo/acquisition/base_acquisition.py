from typing import NoReturn

from torch import Tensor

from ..gp import GP


class BaseAcquisition:

    def __init__(self, model: GP) -> None:
        self.model = model

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> NoReturn:
        raise NotImplementedError('BaseAcquisition::forward()')
