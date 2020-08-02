import pytest

from gpybo.utils.shaping import *


@pytest.mark.parametrize('x', [torch.tensor(0.0), torch.tensor([0.0]), torch.tensor([[0.0]])])
def test_uprank_two(x: Tensor) -> None:

    @uprank_two
    def retval(x_in: Tensor) -> Tensor:
        return x_in

    ret = retval(x)

    assert len(ret.shape) == 2
    assert isinstance(ret, Tensor)


@pytest.mark.parametrize('n', [3, 5, 10])
def test_uprank(n: int):

    x = torch.tensor(0.0)
    ret = uprank(x, n)

    assert len(ret.shape) == n


@pytest.mark.parametrize('x', [torch.rand(1, 1, 1), torch.rand(1, 1, 1, 1)])
def test_uprank_ve(x: Tensor) -> None:

    with pytest.raises(ValueError):
        ret = uprank(x, 2)


@pytest.mark.parametrize('x', [0, 0.0, [0.0, 0.0], np.array([0.0, 0.0])])
def test_to_tensor(x: Any) -> None:

    @to_tensor
    def retval(x_in: Any) -> Tensor:
        return x_in

    ret = retval(x)

    assert isinstance(ret, Tensor)


@pytest.mark.parametrize('x', [0, 0.0, [0.0, 0.0], np.array([0.0, 0.0])])
def test_convert_tensor(x: Any) -> None:

    ret = convert_tensor(x)

    assert isinstance(ret, Tensor)


@pytest.mark.parametrize('x', [0, 0.0, [0.0, 0.0], torch.tensor([0.0, 0.0])])
def test_to_array(x: Any) -> None:

    @to_array
    def retval(x_in: Any) -> Tensor:
        return x_in

    ret = retval(x)

    assert isinstance(ret, np.ndarray)


@pytest.mark.parametrize('x', [0, 0.0, [0.0, 0.0], torch.tensor([0.0, 0.0])])
def test_convert_array(x: Any) -> None:

    ret = convert_array(x)

    assert isinstance(ret, np.ndarray)


def test_unwrap() -> None:

    x1 = torch.ones(5, 3)
    x2 = torch.tensor([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ])

    expected1 = torch.ones(15, 1)
    expected2 = torch.arange(1, 10).view(-1, 1)

    ret1 = unwrap(x1)
    ret2 = unwrap(x2)

    assert torch.all(torch.eq(ret1, expected1))
    assert torch.all(torch.eq(ret2, expected2))
