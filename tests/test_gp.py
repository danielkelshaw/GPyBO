import pytest
import torch
from torch import Tensor

from gpybo.gp import GP
from gpybo.kernel import SquaredExponentialKernel


class TestGP:

    @pytest.fixture
    def gp(self):

        kern = SquaredExponentialKernel()
        gp = GP(kernel=kern)

        return gp

    def test_predictive_posterior(self, gp):

        x = torch.tensor([-4, -3, -2, -1, 1], dtype=torch.float32)
        y = torch.sin(x)

        xp = torch.arange(-5, 5, 0.5)

        gp = gp | (x, y)
        mu_s, cov_s = gp.predictive_posterior(xp)

        assert mu_s.shape == (20, 1)
        assert cov_s.shape == (20, 20)

        assert isinstance(mu_s, Tensor)
        assert isinstance(cov_s, Tensor)

    def test_observe(self, gp):

        x = torch.tensor([-4, -3, -2, -1, 1], dtype=torch.float32)
        y = torch.sin(x)

        gp.observe(x, y)

        assert torch.allclose(gp.x, x.view_as(gp.x))
        assert torch.allclose(gp.y, y.view_as(gp.y))
