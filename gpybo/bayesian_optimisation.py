from typing import Any, Callable, List, NoReturn, Tuple

import torch
from torch import Tensor

from .gp import GP
from .acquisition import qExpectedImprovement
from .utils.samplers import draw_sobol
from .utils.optim import scipy_acqopt


class BO:

    def __init__(self,
                 model: GP,
                 lb: Tensor,
                 ub: Tensor,
                 fn: Callable[[Any], Any]) -> None:

        """Bayesian Optimisation.

        Parameters
        ----------
        model : GP
            Gaussian Process Model.
        lb : Tensor
            Lower bounds.
        ub : Tensor
            Upper bounds.
        fn : function
            Function to optimise.
        """

        self.model = model

        self.lb = lb
        self.ub = ub

        self.acquisition = qExpectedImprovement
        self.opt_acquisition = torch.optim.Adam

        self.fn = fn

    def _optimise_acquisition(self, n_samples: int = 100) -> Tuple[Tensor, float]:

        """Optimises the Acquisition Function.

        Parameters
        ----------
        n_samples : int
            Number of initial samples to draw.

        Returns
        -------
        xopt : Tensor
            X-value which maximises the acquisition function.
        """

        acquisition = self.acquisition(self.model)

        xcandidates = draw_sobol(torch.stack((self.lb, self.ub), -1), n_samples)
        acqcandidates = []

        for x in xcandidates:
            try:
                acq = acquisition(x.unsqueeze(0))
                acqcandidates.append(acq)
            except ValueError:
                acqcandidates.append(0.0)

        acqcandidates = torch.tensor(acqcandidates)
        xopt = xcandidates[acqcandidates.argmax()].unsqueeze(0)

        xopt, loss = scipy_acqopt(xopt, self.lb, self.ub, acquisition)

        return xopt, loss

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

            x, loss = self._optimise_acquisition()

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

            print(f'Observation {i:02}')

            x_sample, _ = self.optimise()

            self.model.observe(x_sample, self.fn(x_sample))
            curr_loss = self.model.train()
