from typing import NoReturn


class NewBO:

    def __init__(self) -> None:
        pass

    def acq_optimise(self) -> NoReturn:
        raise NotImplementedError('NewBO::acq_optimise()')

    def optimise(self) -> NoReturn:
        raise NotImplementedError('NewBO::optimise()')
