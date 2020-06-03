import torch
from torch import Tensor

from gpybo.kernel import SquaredExponentialKernel


class TestSquaredExponentialKernel:

    def test_calculate(self):
        kern = SquaredExponentialKernel()

        x = torch.arange(1, 4, dtype=torch.float32)
        xp = torch.arange(1, 4, dtype=torch.float32)

        target = torch.tensor([[1.0, 0.60653066, 0.13533528],
                               [0.60653066, 1.0, 0.60653066],
                               [0.13533528, 0.60653066, 1.0]],
                              dtype=torch.float32)

        ret_kern = kern.calculate(x, xp)

        assert isinstance(ret_kern, Tensor)
        assert torch.allclose(ret_kern, target, atol=1e-3)

    def test_covariance(self):
        kern = SquaredExponentialKernel()

        x = torch.arange(1, 5, dtype=torch.float32)
        xp = torch.arange(5, 7, dtype=torch.float32)

        target = torch.tensor([[1.000, 0.607, 0.135, 0.011, 0.000, 0.000],
                               [0.607, 1.000, 0.607, 0.135, 0.011, 0.000],
                               [0.135, 0.607, 1.000, 0.607, 0.135, 0.011],
                               [0.011, 0.135, 0.607, 1.000, 0.607, 0.135],
                               [0.000, 0.011, 0.135, 0.607, 1.000, 0.607],
                               [0.000, 0.000, 0.011, 0.135, 0.607, 1.000]],
                              dtype=torch.float32)

        ret_kern = kern.covariance(x, xp)

        assert isinstance(ret_kern, Tensor)
        assert torch.allclose(ret_kern, target, atol=1e-3)
