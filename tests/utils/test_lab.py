import torch
from gpybo.utils.lab import pw_dist2


class TestLab:

    def test_pw_dist2(self):

        x = torch.tensor([0, 1, 2], dtype=torch.float32)
        dst = pw_dist2(x, x)
        target = torch.tensor([
            [0, 1, 4],
            [1, 0, 1],
            [4, 1, 0]
        ], dtype=torch.float32)

        assert torch.allclose(dst, target, atol=1e-5)
