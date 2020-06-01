import pytest
import torch

from gpybo.likelihood import GaussianLikelihood
from gpybo.kernel import SquaredExponentialKernel


class TestGaussianLikelihood:

    @pytest.fixture
    def gaussian_likelihood(self):
        return GaussianLikelihood()

    def test_log_likelihood(self, gaussian_likelihood):

        kernel = SquaredExponentialKernel()
        x = torch.tensor([-4, -3, -2, -1], dtype=torch.float32)
        y = torch.sin(x)

        ll = gaussian_likelihood.log_likelihood(kernel, x, y)
        target = torch.tensor([-3.2324, -2.8482, -3.0372, -3.0269],
                              dtype=torch.float32)

        assert torch.allclose(ll, target, atol=1e-3)
