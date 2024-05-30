import numpy as np

from ncs.propagators.shift import Shift


class TestShift:
    def test___str__(self) -> None:
        assert str(Shift([2, 1, 0, 3], [10, 11])) == "[2 1]=[0 3]+[10 11]"

    def test_compute_domains(self) -> None:
        domains = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])
        variables = [2, 1, 0, 3]
        constants = [5, 4]
        assert np.all(
            Shift(variables, constants).compute_domains(domains) == np.array([[5, 13], [4, 12], [-5, 3], [-4, 4]])
        )
