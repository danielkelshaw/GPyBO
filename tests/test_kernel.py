import numpy as np

from gpybo.kernel import SquaredExponentialKernel


class TestSquaredExponentialKernel:

    def test_calculate(self):
        kern = SquaredExponentialKernel()

        x = np.arange(1, 4)
        xp = np.arange(1, 4)

        target = np.array([[1.0, 0.60653066, 0.13533528],
                           [0.60653066, 1.0, 0.60653066],
                           [0.13533528, 0.60653066, 1.0]])

        xp_mesh, x_mesh = np.meshgrid(xp, x)
        ret_kern = kern.calculate(x_mesh, xp_mesh)

        assert isinstance(ret_kern, np.ndarray)
        assert np.allclose(ret_kern, target, rtol=1e-5)

    def test_calculate_kernel(self):
        kern = SquaredExponentialKernel()

        x = np.arange(1, 4)
        xp = np.arange(1, 4)

        target = np.array([[1.0, 0.60653066, 0.13533528],
                           [0.60653066, 1.0, 0.60653066],
                           [0.13533528, 0.60653066, 1.0]])

        ret_kern = kern.calculate_kernel(x, xp)

        assert isinstance(ret_kern, np.ndarray)
        assert np.allclose(ret_kern, target, rtol=1e-5)
