import numpy as np
from typing import NoReturn, Tuple


class GP:

    def __init__(self, kernel) -> None:

        self.kernel = kernel

    def log_likelihood(self):
        raise NotImplementedError('GP::log_likelihood()')

    def train(self) -> NoReturn:
        raise NotImplementedError('GP::train()')

    def predictive_posterior(self,
                             x: np.ndarray,
                             xp: np.ndarray,
                             y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k_xx = self.kernel.calculate_kernel(x, x)
        k_xxp = self.kernel.calculate_kernel(x, xp)
        k_xpx = self.kernel.calculate_kernel(xp, x)
        k_xpxp = self.kernel.calculate_kernel(xp, xp)

        k_xx_inv = np.linalg.inv(k_xx)

        p_mean = np.matmul(np.matmul(k_xpx, k_xx_inv), y)
        p_covariance = k_xpxp - np.matmul(np.matmul(k_xpx, k_xx_inv), k_xxp)

        return p_mean, p_covariance
