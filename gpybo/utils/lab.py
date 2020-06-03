import torch
from torch import Tensor

from .uprank import uprank


@uprank
def pw_dist2(x: Tensor, xp: Tensor) -> Tensor:

    norm_x = torch.sum(x ** 2, dim=1)[:, None]
    norm_xp = torch.sum(xp ** 2, dim=1)[None, :]

    sq_dst = norm_x + norm_xp - 2 * x @ torch.transpose(xp, 0, 1)

    return sq_dst


@uprank
def pw_dist(x: Tensor, xp: Tensor) -> Tensor:

    sq_dst = pw_dist2(x, xp)
    dst = torch.sqrt(torch.max(sq_dst, torch.tensor(1e-16)))

    return dst
