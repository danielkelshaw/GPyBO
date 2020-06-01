import numpy as np
from typing import Any, NoReturn, Sequence, Tuple
from .kernel import Kernel


class GP:

    def __init__(self, kernel: Kernel) -> None:

        self.kernel = kernel
        self.x = None
        self.y = None

    def log_likelihood(self) -> NoReturn:
        raise NotImplementedError('GP::log_likelihood()')

    def train(self) -> NoReturn:
        raise NotImplementedError('GP::train()')

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def predictive_posterior(self, xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k_xx = self.kernel.calculate_kernel(self.x, self.x)
        k_xxp = self.kernel.calculate_kernel(self.x, xp)
        k_xpx = self.kernel.calculate_kernel(xp, self.x)
        k_xpxp = self.kernel.calculate_kernel(xp, xp)

        k_xx_inv = np.linalg.inv(k_xx)

        p_mean = np.matmul(np.matmul(k_xpx, k_xx_inv), self.y)
        p_covariance = k_xpxp - np.matmul(np.matmul(k_xpx, k_xx_inv), k_xxp)

        return p_mean, p_covariance

    def __or__(self, other: Sequence[Any]) -> 'GP':

        if not len(other) == 2:
            raise ValueError('Must provide (x, y)')

        self.observe(*other)
        return self
