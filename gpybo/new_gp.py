import torch.nn as nn

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .new_kernel.base_kernel import BaseKernel
from .new_mean.base_mean import BaseMean


"""
At the moment the implementation of `GP` is rather messy and the inputs
are not consistent - this is leading to troubles with implementation of
Bayesian Optimisation.

This stage of work aims to focus on improving the quality of the GP
code such that it is not a blocker for the BO project.


// Stage One:
- [ ] Get everything working with 'hardcoded' shapes.
- [ ] Implement additional tests to make sure these work as intended.

// Stage Two:
- [ ] Add decorators to allow less constrained input (for users).
- [ ] Implement tests for decorators to ensure they work as intended.
- [ ] Add user-friendly interfaces such as __call__ and __repr__.
"""


class NewGP(nn.Module):

    def __init__(self, kernel: BaseKernel, mean: BaseMean) -> None:

        super().__init__()

        self.mean = mean
        self.kernel = kernel

        self.x = None
        self.y = None

    def observe(self, x: Tensor, y: Tensor) -> None:
        raise NotImplementedError('NewGP::observe()')

    def posterior(self, xp: Tensor) -> MultivariateNormal:
        raise NotImplementedError('NewGP::posterior()')

    def optimise(self) -> None:
        raise NotImplementedError('NewGP::optimise()')

    def optimise_restarts(self, n_restarts: int) -> None:
        raise NotImplementedError('NewGP::optimise_restarts()')

    def log_likelihood(self) -> Tensor:
        raise NotImplementedError('NewGP::log_likelihood()')
