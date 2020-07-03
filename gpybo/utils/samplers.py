from typing import Optional

from torch import Tensor
from torch.quasirandom import SobolEngine


def draw_sobol(bounds: Tensor, n: int, seed: Optional[int] = None) -> Tensor:

    """Draws samples from Sobol Sequence.

    Parameters
    ----------
    bounds : Tensor
        Bounds between which to draw samples.
    n : int
        Number of samples to draw.
    seed : int
        Seed to use for SobolEngine.

    Returns
    -------
    Tensor
        Samples drawn from Sobol sequence.
    """

    dim = bounds.shape[0]

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    rng = ub - lb

    engine = SobolEngine(dim, scramble=True, seed=seed)
    samples = engine.draw(n, dtype=lb.dtype).view(n, dim)

    return lb + rng * samples
