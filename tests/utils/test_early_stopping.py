import torch
from gpybo.utils.early_stopping import EarlyStopping


class TestEarlyStopping:

    def test_es(self):

        es = EarlyStopping(patience=5, delta=0.0)
        stop = False
        for i in range(15):
            stop = es(loss=torch.tensor(10.0))

        assert stop
