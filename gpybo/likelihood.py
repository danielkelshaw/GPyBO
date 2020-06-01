import numpy as np
from .kernel import Kernel


class Likelihood:

    def __init__(self) -> None:
        pass

    def log_likelihood(self, *args):
        raise NotImplementedError('Likelihood::log_likelihood()')


class GaussianLikelihood(Likelihood):

    def __init__(self) -> None:
        super().__init__()

    def log_likelihood(self, kernel: Kernel, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        K = kernel.calculate_kernel(x, x)
        L = np.linalg.cholesky(K)

        alpha = np.linalg.lstsq(L.T, np.linalg.lstsq(L, y, rcond=None)[0], rcond=None)[0]
        const_term = 0.5 * len(x) * np.log(2 * np.pi)

        ll = -0.5 * y.T * alpha - np.sum(np.log(L.diagonal())) - const_term

        return ll
