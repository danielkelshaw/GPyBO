import torch
from torch import Tensor
from gpybo.utils.shaping import to_tensor, uprank_two


class TestShaping:

    def test_to_tensor(self):

        x = [0, 1, 2]

        @to_tensor
        def ret_val(val):
            return val

        assert isinstance(ret_val(x), Tensor)

    def test_uprank_two(self):

        x = torch.tensor(3.0, dtype=torch.float32)

        @uprank_two
        def ret_val(val):
            return val

        assert ret_val(x).shape == (1, 1)
