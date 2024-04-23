import numpy as np

from ncs.constraints.sum import Sum
from ncs.problem import Problem


class SumTest:
    def test_compute_domains(self) -> None:
        domains = np.array(
            [
                [0, 2],
                [0, 2],
                [4, 6],
            ]
        )
        problem = Problem(domains)
        sum = Sum(np.array([2, 0, 1]))
        assert np.all(
            sum.compute_domains(problem)
            == np.array(
                [
                    [2, 2],
                    [2, 2],
                    [4, 4],
                ]
            )
        )

    def test_filter(self) -> None:
        domains = np.array(
            [
                [0, 2],
                [0, 2],
                [4, 6],
            ]
        )
        problem = Problem(domains)
        sum = Sum(np.array([2, 0, 1]))
        assert np.all(
            sum.filter(problem)
            == np.array(
                [
                    [2, 2],
                    [2, 2],
                    [4, 4],
                ]
            )
        )
