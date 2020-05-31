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
        xp = np.arange(-5, 5, 0.5)
        y = np.sin(x)

        mu_s, cov_s = gp.predictive_posterior(x, xp, y)

        assert mu_s.shape == (20,)
        assert cov_s.shape == (20, 20)

        assert isinstance(mu_s, np.ndarray)
        assert isinstance(cov_s, np.ndarray)
