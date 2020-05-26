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
        assert np.allclose(ret_kern, target, atol=1e-3)

    def test_calculate_kernel(self):
        kern = SquaredExponentialKernel()

        x = np.arange(1, 4)
        xp = np.arange(1, 4)

        target = np.array([[1.0, 0.60653066, 0.13533528],
                           [0.60653066, 1.0, 0.60653066],
                           [0.13533528, 0.60653066, 1.0]])

        ret_kern = kern.calculate_kernel(x, xp)

        assert isinstance(ret_kern, np.ndarray)
        assert np.allclose(ret_kern, target, atol=1e-3)

    def test_covariance(self):
        kern = SquaredExponentialKernel()

        x = np.arange(1, 5)
        xp = np.arange(5, 7)

        target = np.array([[1.000, 0.607, 0.135, 0.011, 0.000, 0.000],
                           [0.607, 1.000, 0.607, 0.135, 0.011, 0.000],
                           [0.135, 0.607, 1.000, 0.607, 0.135, 0.011],
                           [0.011, 0.135, 0.607, 1.000, 0.607, 0.135],
                           [0.000, 0.011, 0.135, 0.607, 1.000, 0.607],
                           [0.000, 0.000, 0.011, 0.135, 0.607, 1.000]])

        ret_kern = kern.covariance(x, xp)

        assert isinstance(ret_kern, np.ndarray)
        assert np.allclose(ret_kern, target, atol=1e-3)
