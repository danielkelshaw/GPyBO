from typing import Any, Callable, Tuple

import torch
from torch import Tensor

from .gp import GP
from .acquisition.analytic import ExpectedImprovement
from .utils.samplers import draw_sobol
from .utils.optim import scipy_acqopt


class BO:

    def __init__(self, model: GP, lb: Tensor, ub: Tensor) -> None:

        """Bayesian Optimisation

        Parameters
        ----------
        model : GP
            Model to use for Bayesian Optimisation.
        lb : Tensor
            Lower bound of search space.
        ub : ub
            Upper bound of search space.
        """

        self.model = model

        self.lb = lb
        self.ub = ub

        self.acquisition = ExpectedImprovement

    def optimise_acquisition(self, nsamples: int = 512) -> Tuple[Tensor, Tensor]:

        """Optimise the Acquisition Function.

        Parameters
        ----------
        nsamples : int
            Number of Sobol samples to use.

        Returns
        -------
        xnew : Tensor
            Point which maximises the acquisition function.
        loss : Tensor
            Value of the acquisition function.
        """

        acq = self.acquisition(self.model)
        x_draw = draw_sobol(self.lb, self.ub, n=nsamples)
        acq_draw = acq(x_draw)

        xopt = x_draw[acq_draw.argmax()].unsqueeze(0)
        xnew, loss = scipy_acqopt(xopt, self.lb, self.ub, acq)

        return xnew.unsqueeze(0), loss

    def optimise_acquisition_restarts(self, n_restarts: int = 10, nsamples: int = 512) -> Tuple[Tensor, Tensor]:

        """Optimise Acquisition Function with Restarts.

        Parameters
        ----------
        n_restarts : int
            Number of restarts to use in optimisation.
        nsamples : int
            Number of Sobol samples to use.

        Returns
        -------
        Tensor
            Candidate which maximises the acquisition function.
        Tensor
            Maximum value of the acquisition function.
        """

        losses = []
        candidates = []

        for i in range(n_restarts):

            _x, _loss = self.optimise_acquisition(nsamples)

            losses.append(_loss)
            candidates.append(_x)

        losses = torch.cat(losses)
        candidates = torch.cat(candidates)

        best = losses.argmin()

        return candidates[best].unsqueeze(0), losses[best]

    def optimise_fn(self, fn: Callable[[Any], Any], n_iterations: int) -> Tuple[Tensor, Tensor]:

        """Optimises a Function.

        Parameters
        ----------
        fn : Callable[[Any], Any]
            Function to optimise for.
        n_iterations : int
            Number of iterations to run for.

        Returns
        -------
        Tensor
            Optimal x value.
        Tensor
            Optimal y value.
        """

        for i in range(n_iterations):

            xnew, loss = self.optimise_acquisition()
            self.model.observe(xnew, fn(xnew))
            self.model.optimise()

        idx_max = self.model.y.argmax()

        return self.model.x[idx_max], self.model.y[idx_max]
