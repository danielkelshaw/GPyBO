import numpy as np
import pytest

from gpybo.gp import GP
from gpybo.kernel import SquaredExponentialKernel


class TestGP:

    @pytest.fixture
    def gp(self):

        kern = SquaredExponentialKernel()
        gp = GP(kernel=kern)

        return gp

    def test_predictive_posterior(self, gp):

        x = np.array([-4, -3, -2, -1, 1])
        y = np.sin(x)

        xp = np.arange(-5, 5, 0.5)

        gp = gp | (x, y)
        mu_s, cov_s = gp.predictive_posterior(xp)

        assert mu_s.shape == (20,)
        assert cov_s.shape == (20, 20)

        assert isinstance(mu_s, np.ndarray)
        assert isinstance(cov_s, np.ndarray)

    def test_observe(self, gp):

        x = np.array([-4, -3, -2, -1, 1])
        y = np.sin(x)

        gp.observe(x, y)

        assert np.array_equal(gp.x, x)
        assert np.array_equal(gp.y, y)
