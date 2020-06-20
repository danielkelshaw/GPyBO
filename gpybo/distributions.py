from torch import Tensor
from typing import Optional


class Normal:

    def __init__(self, mu: Optional[Tensor] = None, covariance: Optional[Tensor] = None) -> None:
        self._mu = mu
        self._covariance = covariance

    @property
    def mu(self) -> Tensor:
        return self._mu

    @mu.setter
    def mu(self, mu: Tensor) -> None:
        self._mu = mu

    @property
    def covariance(self) -> Tensor:
        return self._covariance

    @covariance.setter
    def covariance(self, covariance: Tensor) -> None:
        self._covariance = covariance

    @property
    def variance(self) -> Tensor:
        return self.covariance.diag()
