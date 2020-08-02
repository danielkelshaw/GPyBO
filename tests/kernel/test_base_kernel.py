import torch
from torch import Tensor

import numpy as np
import pytest

from gpybo.kernel.base_kernel import SumKernel, ProductKernel, OneKernel
from gpybo.kernel.kernels import SquaredExponentialKernel, Matern32Kernel


class TestBaseKernel:

    def test_add(self) -> None:

        kern = SquaredExponentialKernel() + Matern32Kernel()

        assert isinstance(kern, SumKernel)
        assert len(kern.kernels) == 2

    def test_sub(self) -> None:

        kern = SquaredExponentialKernel() - Matern32Kernel()

        assert isinstance(kern, SumKernel)
        assert len(kern.kernels) == 2

    def test_mul(self) -> None:

        kern = SquaredExponentialKernel() * Matern32Kernel()

        assert isinstance(kern, ProductKernel)
        assert len(kern.kernels) == 2


class TestSumKernel:

    def test_init(self) -> None:

        kern = SumKernel(SquaredExponentialKernel(), Matern32Kernel())
        assert len(kern.kernels) == 2

    def test_add(self) -> None:

        kern = SumKernel()
        kern.add(SquaredExponentialKernel(), Matern32Kernel())

        assert len(kern.kernels) == 2

    def test_add_ve(self) -> None:

        kern = ProductKernel()

        with pytest.raises(ValueError):
            kern.add(np.array([1.0]), SquaredExponentialKernel())

    def test_calculate(self) -> None:

        k_sum = SumKernel(SquaredExponentialKernel(), Matern32Kernel())
        k_sek = SquaredExponentialKernel()
        k_mat = Matern32Kernel()

        x = torch.rand(3, 1)

        k_sum_ret = k_sum.calculate(x, x)
        k_sek_ret = k_sek.calculate(x, x)
        k_mat_ret = k_mat.calculate(x, x)

        assert isinstance(k_sum_ret, Tensor)
        assert torch.allclose(k_sum_ret, k_sek_ret + k_mat_ret)
        assert k_sum_ret.shape == (3, 3)


class TestProductKernel:

    def test_init(self) -> None:

        kern = ProductKernel(SquaredExponentialKernel(), Matern32Kernel())
        assert len(kern.kernels) == 2

    def test_add(self) -> None:

        kern = ProductKernel()
        kern.add(SquaredExponentialKernel(), Matern32Kernel())

        assert len(kern.kernels) == 2

    def test_add_ve(self) -> None:

        kern = ProductKernel()

        with pytest.raises(ValueError):
            kern.add(np.array([1.0]), SquaredExponentialKernel())

    def test_calculate(self) -> None:

        k_prod = ProductKernel(SquaredExponentialKernel(), Matern32Kernel())
        k_sek = SquaredExponentialKernel()
        k_mat = Matern32Kernel()

        x = torch.rand(3, 1)

        k_prod_res = k_prod.calculate(x, x)
        k_sek_res = k_sek.calculate(x, x)
        k_mat_res = k_mat.calculate(x, x)

        assert isinstance(k_prod_res, Tensor)
        assert torch.allclose(k_prod_res, k_sek_res * k_mat_res)
        assert k_prod_res.shape == (3, 3)


class TestOneKernel:

    def test_calculate(self) -> None:

        x1 = torch.rand(5, 1)
        x2 = torch.rand(3, 1)

        kern = OneKernel()
        ret = kern.calculate(x1, x2)

        assert isinstance(ret, Tensor)
        assert ret.shape == (5, 3)
