from scipy.optimize import Bounds, minimize
from .shaping import to_array, convert_array, convert_tensor

import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from ..acquisition import BaseAcquisitionFunction


def scipy_acqopt(x0: Tensor, lb: Tensor, ub: Tensor,
                 acq: BaseAcquisitionFunction) -> Tuple[Tensor, Tensor]:

    _x, _lb, _ub = (convert_array(x) for x in [x0, lb, ub])
    bounds = Bounds(_lb, _ub)

    def f(x: np.ndarray) -> Tuple[float, np.ndarray]:
        X = torch.from_numpy(x).view(x0.shape).contiguous().requires_grad_(True)
        loss = -acq(X).sum()
        grad = convert_array(torch.autograd.grad(loss, X)[0].contiguous())

        return loss.item(), grad

    res = minimize(
        fun=f,
        x0=_x,
        jac=True,
        bounds=bounds,
    )

    return convert_tensor(res.x), convert_tensor(res.fun).unsqueeze(0)
