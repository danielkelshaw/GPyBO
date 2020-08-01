from gpybo.mean.base_mean import BaseMean


class TestBaseMean:

    def test_len(self) -> None:

        mean = BaseMean()
        assert len(mean) == 1
