import torch
from torch import Tensor

from gpybo.mean.means import ZeroMean


class TestZeroMean:

    def test_calculate(self) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])
        expected = torch.zeros(3, 1)

        mean = ZeroMean()
        ret = mean.calculate(x)

        assert isinstance(ret, Tensor)
        assert ret.shape == x.shape
        assert torch.allclose(ret, expected)
