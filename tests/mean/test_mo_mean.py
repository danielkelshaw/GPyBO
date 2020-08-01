import pytest
import torch
from torch import Tensor
from gpybo.mean.means import ZeroMean
from gpybo.mean.mo_mean import MOMean


class TestMOMean:

    @pytest.fixture
    def zm_zm_mean(self) -> MOMean:
        return MOMean([ZeroMean(), ZeroMean()])

    def test_init(self) -> None:

        means = [ZeroMean(), ZeroMean()]
        mean = MOMean(means)

        assert len(mean.means) == 2

    def test_init_ve(self) -> None:

        means = [1.0, 2.0]

        with pytest.raises(TypeError):
            mean = MOMean(means)

    def test_getitem(self, zm_zm_mean) -> None:

        assert isinstance(zm_zm_mean[0], ZeroMean)
        assert isinstance(zm_zm_mean[1], ZeroMean)

    def test_len(self, zm_zm_mean) -> None:

        assert len(zm_zm_mean) == 2

    def test_calculate(self, zm_zm_mean) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])
        ret = zm_zm_mean.calculate(x)

        assert ret.shape == (x.shape[0] * 2, 1)
        assert isinstance(ret, Tensor)
        assert torch.allclose(ret, torch.zeros(6, 1))
