import pytest

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from gpybo.gp import GP
from gpybo.kernel.kernels import SquaredExponentialKernel
from gpybo.kernel.mo_kernel import MOKernel


class TestGP:

    @pytest.fixture
    def init_gp(self) -> GP:
        return GP(SquaredExponentialKernel())

    @pytest.fixture
    def obs_gp(self) -> GP:

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = torch.tensor([[0.0], [1.0], [2.0]])

        gp = GP(SquaredExponentialKernel())
        gp.observe(x, y)

        return gp

    def test_init_ve(self) -> None:

        mo_kernel = MOKernel([
            SquaredExponentialKernel(),
            SquaredExponentialKernel(),
            SquaredExponentialKernel()
        ])

        with pytest.raises(ValueError):
            gp = GP(kernel=mo_kernel)

    def test_observe(self, init_gp) -> None:

        x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y1 = torch.tensor([[0.0], [1.0], [2.0]])

        init_gp.observe(x1, y1)

        assert torch.allclose(init_gp.x, x1)
        assert torch.allclose(init_gp.y, y1)

        x2 = torch.tensor([[3.0, 3.0]])
        y2 = torch.tensor([[3.0]])

        init_gp.observe(x2, y2)

        assert torch.allclose(init_gp.x, torch.cat((x1, x2), dim=0))
        assert torch.allclose(init_gp.y, torch.cat((y1, y2), dim=0))

    def test_or(self, init_gp) -> None:

        x_in = [0.0, 1.0, 2.0]
        y_in = [0.0, 1.0, 2.0]

        x_target = torch.tensor([[0.0], [1.0], [2.0]])
        y_target = torch.tensor([[0.0], [1.0], [2.0]])

        gp = init_gp | (x_in, y_in)

        assert torch.all(torch.eq(gp.x, x_target))
        assert torch.all(torch.eq(gp.y, y_target))

    def test_or_ve(self, init_gp) -> None:

        x_in = [0.0, 1.0, 2.0]
        y_in = [0.0, 1.0, 2.0]
        z_in = [0.0, 1.0, 2.0]

        with pytest.raises(ValueError):
            gp = init_gp | (x_in, y_in, z_in)

    def test_observe_ve(self, init_gp) -> None:

        x1 = torch.tensor([[0.0, 1.0, 2.0]])
        y1 = torch.tensor([[0.0], [1.0]])

        with pytest.raises(ValueError):
            init_gp.observe(x1, y1)

    def test_posterior(self, obs_gp) -> None:

        xp1 = torch.tensor([[3.0, 3.0]])
        xp2 = torch.tensor([[3.0, 3.0], [4.0, 4.0]])

        p1 = obs_gp.posterior(xp1)
        p2 = obs_gp.posterior(xp2)

        assert isinstance(p1, MultivariateNormal)
        assert isinstance(p2, MultivariateNormal)

        assert p1.mean.shape == (1,)
        assert p2.mean.shape == (2,)

    def test_posterior_ve(self, obs_gp) -> None:

        xp = torch.tensor([3.0, 3.0])

        with pytest.raises(ValueError):
            p = obs_gp.posterior(xp)

    def test_call(self) -> None:

        x = torch.tensor([[0.0], [1.0], [2.0]])
        y = torch.tensor([[0.0], [1.0], [2.0]])

        gp = GP(SquaredExponentialKernel()) | (x, y)

        x_call = 3.0
        x_post = torch.tensor([[3.0]])

        p_call = gp(x_call)
        p_post = gp(x_post)

        assert torch.allclose(p_call.mean, p_post.mean)
        assert p_call.mean.shape == p_post.mean.shape

    def test_call_ve(self, obs_gp) -> None:

        x_call = [3.0, 3.0]

        with pytest.raises(ValueError):
            p_call = obs_gp(x_call)

    def test_optimise(self, obs_gp) -> None:

        loss = obs_gp.optimise(n_iterations=100)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()

    def test_optimise_restarts(self, obs_gp) -> None:

        loss = obs_gp.optimise_restarts(n_restarts=3, n_iterations=10)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()

    def test_log_likelihood(self, obs_gp) -> None:

        ll = obs_gp.log_likelihood(grad=False)
        ll_grad = obs_gp.log_likelihood(grad=True)

        assert isinstance(ll, Tensor)
        assert isinstance(ll_grad, Tensor)
        assert ll.shape == ll_grad.shape == ()

        with pytest.raises(RuntimeError):
            ll.backward()
