from gpybo.kernel.kernels import *


class TestZeroKernel:

    def test_calculate(self) -> None:

        x1 = torch.rand(5, 1)
        x2 = torch.rand(3, 1)

        kern = ZeroKernel()
        ret = kern.calculate(x1, x2)

        assert isinstance(ret, Tensor)
        assert ret.shape == (5, 3)
        assert torch.allclose(ret, torch.zeros(5, 3))


class TestSquaredExponentialKernel:

    def test_calculate(self) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])
        expected = torch.tensor([
            [1.0000, 0.6065, 0.1353],
            [0.6065, 1.0000, 0.6065],
            [0.1353, 0.6065, 1.0000]
        ])

        kern = SquaredExponentialKernel()
        ret = kern.calculate(x, x)

        assert isinstance(ret, Tensor)
        assert ret.shape == (x.shape[0], x.shape[0])
        assert torch.allclose(ret, expected, atol=1e-4)


class TestSincKernel:

    def test_calculate(self) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])
        xp = torch.tensor([[1.5], [2.5], [3.5]])

        expected = torch.tensor([
            [0.9589, 0.6650, 0.2394],
            [0.9589, 0.9589, 0.6650],
            [0.6650, 0.9589, 0.9589]
        ])

        kern = SincKernel()
        ret = kern.calculate(x, xp)

        assert isinstance(ret, Tensor)
        assert ret.shape == (x.shape[0], xp.shape[0])
        assert torch.allclose(ret, expected, atol=1e-4)


class TestMatern32Kernel:

    def test_calculate(self) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])

        expected = torch.tensor([
            [1.0000, 0.7358, 0.4060],
            [0.7358, 1.0000, 0.7358],
            [0.4060, 0.7358, 1.0000]
        ])

        kern = Matern32Kernel()
        ret = kern.calculate(x, x)

        assert isinstance(ret, Tensor)
        assert ret.shape == (x.shape[0], x.shape[0])
        assert torch.allclose(ret, expected, atol=1e-4)


class TestMatern52Kernel:

    def test_calculate(self) -> None:

        x = torch.tensor([[1.0], [2.0], [3.0]])

        expected = torch.tensor([
            [1.0000, 0.5240, 0.1387],
            [0.5240, 1.0000, 0.5240],
            [0.1387, 0.5240, 1.0000]
        ])

        kern = Matern52Kernel()
        ret = kern.calculate(x, x)

        assert isinstance(ret, Tensor)
        assert ret.shape == (x.shape[0], x.shape[0])
        assert torch.allclose(ret, expected, atol=1e-4)