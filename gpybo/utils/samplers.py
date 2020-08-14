from typing import Optional

from torch import Tensor
from torch.quasirandom import SobolEngine


def draw_sobol(lb: Tensor, ub: Tensor, n: int, seed: Optional[int] = None) -> Tensor:

    """Draws samples from Sobol Sequence.

    Parameters
    ----------
    lb : Tensor
        Lower bound.
    ub : Tensor
        Upper bound.
    n : int
        Number of samples to draw.
    seed : int
        Seed to use for SobolEngine.

    Returns
    -------
    Tensor
        Samples drawn from Sobol sequence.
    """

    dim = lb.shape[0]

    engine = SobolEngine(dim, scramble=True, seed=seed)
    samples = engine.draw(n, dtype=lb.dtype).view(n, dim)

    return lb + (ub - lb) * samples
