import numpy as np

from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problem import Problem


class TestMinValueHeuristic:
    def test_make_value_choice(self) -> None:
        domains = np.array([[0, 0], [0, 2]])
        problem = Problem(domains)
        assert np.all(MinValueHeuristic().make_value_choice(problem, 1) == np.array([[0, 0], [1, 2]]))
