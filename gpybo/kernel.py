import numpy as np


class Kernel:

    def __init__(self) -> None:
        pass

    def calculate(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Kernel::calculate()')

    def calculate_kernel(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:

        """Produces the Kernel given two sets of random variables.

        The calculation of the kernel depends on the kernel method being
        used - this is defined within the `self.calculate()` method. In
        order to calculate the kernel as efficiently as possible the RVs
        are converted to a pair of meshed grids - this ensures that the
        kernel calculation remains simple.

        Parameters
        ----------
        x : np.ndarray
            First set of random variables.
        xp : np.ndarray
            Second set of random variables.

        Returns
        -------
        np.ndarray
            Calculated kernel.
        """

        return self.calculate(*np.meshgrid(xp, x))

    def covariance(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:

        """Calculates the Covariance Matrix.

        Parameters
        ----------
        x : np.ndarray
            First set of random variables.
        xp : np.ndarray
            Second set of random variables.

        Returns
        -------
        covariance : np.ndarray
            Calculated covariance matrix.
        """

        if not x.ndim == xp.ndim == 1:
            raise AssertionError('x and xp must be one-dimensional.')

        n = x.size + xp.size
        covariance = np.zeros((n, n))

        covariance[0:x.size, 0:x.size] = self.calculate_kernel(x, x)
        covariance[0:x.size, x.size:] = self.calculate_kernel(x, xp)
        covariance[x.size:, :x.size] = self.calculate_kernel(xp, x)
        covariance[x.size:, x.size:] = self.calculate_kernel(xp, xp)

        return covariance


class SquaredExponentialKernel(Kernel):

    def __init__(self) -> None:

        super().__init__()

        self.l = 1.0
        self.sigma = 1.0

    def calculate(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:

        """Squared Exponential Kernel Calculation.

        Function expects the two inputs: x, xp to have the same shape.
        This ensures that matrix operations can be carried out without
        any issues.

        Parameters
        ----------
        x : np.ndarray
            First set of random variables.
        xp : np.ndarray
            Second set of random variables.

        Returns
        -------
        np.ndarray
            Calculated kernel.
        """

        if not x.shape == xp.shape:
            msg = f'x and xp must have the same shape - received: '
            msg += f'x with shape {x.shape} and '
            msg += f'xp with shape {xp.shape}'
            raise AssertionError(msg)

        return self.l ** 2 * np.exp(-0.5 * np.square((x - xp) / self.sigma))
