import pytest
import torch
from torch import Tensor

from gpybo.kernel.kernels import SquaredExponentialKernel, Matern32Kernel
from gpybo.kernel.mo_kernel import MOKernel


class TestMOKernel:

    @pytest.fixture
    def sek_mat_kern(self) -> MOKernel:
        return MOKernel([SquaredExponentialKernel(), Matern32Kernel()])

    def test_init(self) -> None:

        kern = MOKernel([SquaredExponentialKernel(), Matern32Kernel()])
        assert len(kern.kernels) == 2

    def test_init_ve(self) -> None:

        with pytest.raises(TypeError):
            kern = MOKernel([0.0, 1.0])

    def test_getiterm(self, sek_mat_kern) -> None:

        assert isinstance(sek_mat_kern[0], SquaredExponentialKernel)
        assert isinstance(sek_mat_kern[1], Matern32Kernel)

    def test_len(self, sek_mat_kern) -> None:

        assert len(sek_mat_kern) == 2

    def test_calculate(self, sek_mat_kern) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])
        expected = torch.tensor([
            [1.0000, 0.6065, 0.1353, 0.0000, 0.0000, 0.0000],
            [0.6065, 1.0000, 0.6065, 0.0000, 0.0000, 0.0000],
            [0.1353, 0.6065, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.7358, 0.4060],
            [0.0000, 0.0000, 0.0000, 0.7358, 1.0000, 0.7358],
            [0.0000, 0.0000, 0.0000, 0.4060, 0.7358, 1.0000]
        ])

        ret = sek_mat_kern.calculate(x, x)

        assert isinstance(ret, Tensor)
        assert ret.shape == (x.shape[0] * 2, x.shape[0] * 2)
        assert torch.allclose(ret, expected, atol=1e-4)
