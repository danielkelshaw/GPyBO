from torch import Tensor


class EarlyStopping:

    def __init__(self, patience: int = 15, delta: float = 0.0) -> None:

        """Early Stopping for ML Training.

        Parameters
        ----------
        patience : int
            Number of iterations to wait for improvement.
        delta : float
            Allowable fluctuation in loss.
        """

        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss: Tensor) -> bool:

        """Determines if training should stop.

        Parameters
        ----------
        loss : Tensor
            Loss used to determine ES condition.

        Returns
        -------
        bool
            True if ES, False otherwise.
        """

        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

        return self.early_stop
