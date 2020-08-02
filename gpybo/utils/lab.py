import torch
from torch import Tensor


def pd_jitter(k: Tensor, max_tries: int = 5) -> Tensor:

    """Make Matrix Positive Definite.

    Parameters
    ----------
    k : Tensor
        Tensor to try and make positive definite.
    max_tries : int
        Number of times to try applying jitter.

    Returns
    -------
    k : Tensor
        Positive Definite Tensor.
    """

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
