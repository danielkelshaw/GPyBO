import torch
from torch import Tensor

from gpybo.gp import GP
from gpybo.kernel import SquaredExponentialKernel
from gpybo.acquisition import ExpectedImprovement


class TestExpectedImprovement:

    def test_forward(self):

        x = torch.tensor([-4, -3, -2, -1, 0, 1], dtype=torch.float32)
        y = torch.sin(x)

        gp = GP(SquaredExponentialKernel()) | (x, y)
        ei = ExpectedImprovement(gp, alpha=0.0)

        xp = torch.tensor(2, dtype=torch.float32)
        ret_val = ei(xp)

        target = torch.tensor([[0.1868]])

        assert isinstance(ret_val, Tensor)
        assert ret_val.shape == (1,)

        assert torch.allclose(ret_val, target, atol=1e-3)