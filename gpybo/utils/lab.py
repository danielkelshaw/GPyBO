import torch
from torch import Tensor

from .shaping import uprank_two


@uprank_two
def pw_dist2(x: Tensor, xp: Tensor) -> Tensor:

    norm_x = torch.sum(x ** 2, dim=1)[:, None]
    norm_xp = torch.sum(xp ** 2, dim=1)[None, :]

    sq_dst = norm_x + norm_xp - 2 * x @ torch.transpose(xp, 0, 1)

    return sq_dst


@uprank_two
def pw_dist(x: Tensor, xp: Tensor) -> Tensor:

    sq_dst = pw_dist2(x, xp)
    dst = torch.sqrt(torch.max(sq_dst, torch.tensor(1e-16)))

    return dst


def pd_jitter(k: Tensor, max_tries: int = 5) -> Tensor:

    if (len(k.shape) != 2) or (k.shape[0] != k.shape[1]):
        raise ValueError('Must be a Square Matrix.')

    tries = 0
    jitter = k.diag().mean() * 1e-6

    while tries < max_tries:

        try:
            L = torch.cholesky(k)
            return k
        except RuntimeError:
            jitter *= 10
            k += jitter * torch.eye(k.shape[0])
            tries += 1

    raise ValueError('Not PD, even with jitter.')
