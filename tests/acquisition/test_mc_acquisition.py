import pytest
import torch
from torch import Tensor

from gpybo.acquisition.mc_acquisition import (
    qExpectedImprovement, qProbabilityOfImprovement, qSimpleRegret
)
from gpybo.gp import GP
from gpybo.kernel.kernels import SquaredExponentialKernel


@pytest.fixture
def model() -> GP:

    x = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 1.0])
    y = torch.sin(x)

    return GP(SquaredExponentialKernel()) | (x, y)


@pytest.fixture
def model_nd() -> GP:

    x = torch.tensor([[-3.0, -3.0], [-2.0, -2.0], [1.0, 1.0], [2.0, 2.0]])
    y = -torch.sum(x ** 2, dim=1).view(-1, 1)

    return GP(SquaredExponentialKernel()) | (x, y)


class TestqExpectedImprovement:

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward(self, model, q) -> None:

        x = torch.tensor([[0.0], [0.5]])

        acq = qExpectedImprovement(model, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward_nd(self, model_nd, q) -> None:

        x = torch.tensor([[0.0, 0.0], [0.5, 0.5]])

        acq = qExpectedImprovement(model_nd, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]


class TestqProbabilityOfImprovement:

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward(self, model, q) -> None:

        x = torch.tensor([[0.0], [0.5]])

        acq = qProbabilityOfImprovement(model, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward_nd(self, model_nd, q) -> None:

        x = torch.tensor([[0.0, 0.0], [0.5, 0.5]])

        acq = qProbabilityOfImprovement(model_nd, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]


class TestqSimpleRegret:

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward(self, model, q) -> None:

        x = torch.tensor([[0.0], [0.5]])

        acq = qSimpleRegret(model, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]

    @pytest.mark.parametrize('q', [1, 5, 10])
    def test_forward_nd(self, model_nd, q) -> None:

        x = torch.tensor([[0.0, 0.0], [0.5, 0.5]])

        acq = qSimpleRegret(model_nd, q)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert res.shape[0] == x.shape[0]
