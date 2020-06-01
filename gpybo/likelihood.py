import numpy as np
import torch
from torch import Tensor

from .kernel import Kernel


class Likelihood:

    def __init__(self) -> None:
        pass

    def log_likelihood(self, *args):
        raise NotImplementedError('Likelihood::log_likelihood()')


class GaussianLikelihood(Likelihood):

    def __init__(self) -> None:
        super().__init__()

    def log_likelihood(self, kernel: Kernel, x: Tensor, y: Tensor) -> Tensor:

        K = kernel.calculate_kernel(x, x)
        L = torch.cholesky(K)

        a0, _ = torch.lstsq(y, L)
        alpha, _ = torch.lstsq(a0, L.T)

        y_alpha = -0.5 * y * alpha.view_as(y)
        trace_log = torch.trace(torch.log(L))
        const = 0.5 * len(x) * np.log(2 * np.pi)

        ll = y_alpha - trace_log - const

        return ll
