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


class SquaredExponentialKernel(Kernel):

    def __init__(self) -> None:
        super().__init__()

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

        return np.exp(-0.5 * np.square(np.abs(x - xp)))
