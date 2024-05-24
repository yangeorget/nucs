import numpy as np

from ncs.propagators.shift import Shift


class TestShift:
    def test___str__(self) -> None:
        variables = np.array([[2, 0], [1, 3]])
        assert str(Shift(variables, np.array([10, 11]))) == "[2 1]=[0 3]+[10 11]"

    def test_compute_domains(self) -> None:
        domains = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])
        variables = np.array([[2, 0], [1, 3]])
        constants = np.array([5, 4])
        assert np.all(
            Shift(variables, constants).compute_domains(domains) == np.array([[-5, 3], [4, 12], [5, 13], [-4, 4]])
        )
