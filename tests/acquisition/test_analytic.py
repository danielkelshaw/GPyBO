import pytest
import torch
from torch import Tensor

from gpybo.acquisition.analytic import (
    ExpectedImprovement, ProbabilityOfImprovement
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


class TestExpectedImprovement:

    def test_forward(self, model) -> None:

        x = torch.tensor([[0.0], [0.5]])
        expected = torch.tensor([0.0100, 0.0368])

        acq = ExpectedImprovement(model)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert torch.allclose(expected, res, atol=1e-4)

    def test_forward_nd(self, model_nd) -> None:

        x = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        expected = torch.tensor([[2.2086, 1.9510]])

        acq = ExpectedImprovement(model_nd)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert torch.allclose(expected, res, atol=1e-4)


class TestProbabilityOfImprovement:

    def test_forward(self, model) -> None:

        x = torch.tensor([[0.0], [0.5]])
        expected = torch.tensor([0.0490, 0.1792])

        acq = ProbabilityOfImprovement(model)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert torch.allclose(expected, res, atol=1e-4)

    def test_forward_nd(self, model_nd) -> None:

        x = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        expected = torch.tensor([0.9919, 0.9995])

        acq = ProbabilityOfImprovement(model_nd)
        res = acq.forward(x)

        assert isinstance(res, Tensor)
        assert torch.allclose(expected, res, atol=1e-4)
