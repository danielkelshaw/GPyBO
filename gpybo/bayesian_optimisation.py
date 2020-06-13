from typing import Any, Callable, List, NoReturn, Tuple

import torch
from torch import Tensor

from .gp import GP
from .acquisition import ExpectedImprovement


class BO:

    def __init__(self,
                 model: GP,
                 bounds: Tensor,
                 fn: Callable[[Any], Any]) -> None:

        """Bayesian Optimisation.

        Parameters
        ----------
        model : GP
            Gaussian Process Model.
        bounds : Tensor
            Lower / Upper Bounds.
        fn : function
            Function to optimise.
        """

        self.model = model

        self.bounds = bounds
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

        self.acquisition = ExpectedImprovement
        self.opt_acquisition = torch.optim.Adam

        self.fn = fn

    def _optimise_acquisition(self,
                              n_iterations: int = 1000) -> Tuple[Tensor, Tensor]:

        """Optimises the Acquisition Function.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to optimise for.

        Returns
        -------
        _x : Tensor
            X-value which maximises the acquisition function.
        loss : Tensor
            Value of the acquisition function.
        """

        _x = self.lb + (self.ub - self.lb) * torch.rand(self.lb.shape)
        _x.requires_grad_(True)

        ei_optimiser = self.opt_acquisition(params=[_x], lr=0.025)
        acquisition = self.acquisition(self.model)

        for i in range(n_iterations):

            loss = -acquisition(_x)

            def closure():
                ei_optimiser.zero_grad()
                loss.backward()
                return loss

            ei_optimiser.step(closure)

        return _x.requires_grad_(False), loss

    def optimise(self,
                 n_restarts: int = 10,
                 n_iterations: int = 1000) -> Tuple[Tensor, Tensor]:

        """Optimises the Acquisition Function with restarts.

        Parameters
        ----------
        n_restarts : int
            Number of times to restart the optimisation process.
        n_iterations : int
            Number of iterations for each optimisation run.

        Returns
        -------
        best_candidate : Tensor
            X-value which maximises the acquisition function.
        best_acquisition : Tensor
            Optimised value of the acquisition function.
        """

        losses: List[Tensor] = []
        candidates: List[Tensor] = []

        for i in range(n_restarts):

            x, loss = self._optimise_acquisition(n_iterations=n_iterations)

            losses.append(loss)
            candidates.append(x)

        losses = torch.cat(losses)
        candidates = torch.cat(candidates)

        best = torch.argmin(losses.view(-1), dim=0)
        best_candidate = candidates[best]
        best_acquisition = losses[best]

        return best_candidate, best_acquisition

    def fn_optimise(self, observation_budget: int = 5) -> None:

        """Optimises self.fn for a given number of observations.

        Parameters
        ----------
        observation_budget : int
            Allowable number of observations to optimise for.
        """

        for i in range(observation_budget):

            x_sample, _ = self.optimise()

            self.model.observe(x_sample, self.fn(x_sample))
            curr_loss = self.model.train()
