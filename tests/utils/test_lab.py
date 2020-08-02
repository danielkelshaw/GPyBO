import pytest

from gpybo.kernel.kernels import SquaredExponentialKernel
from gpybo.utils.lab import *


def test_pd_jitter_ve_sm() -> None:

    k = torch.ones(5, 3)

    with pytest.raises(ValueError):
        ret = pd_jitter(k)


def test_pd_jitter_ve_pd() -> None:

    k = torch.zeros(10, 10)

    with pytest.raises(ValueError):
        ret = pd_jitter(k)


def test_pd_jitter() -> None:

    x = torch.rand(5, 1)
    k = SquaredExponentialKernel()(x, x)

    k_jit = pd_jitter(k)

    assert isinstance(k_jit, Tensor)
    assert k.shape == k_jit.shape
