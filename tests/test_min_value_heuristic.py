import numpy as np

from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import Problem


class TestMinValueHeuristic:
    def test_choose(self) -> None:
        domains = np.array([[0, 0], [0, 2]])
        problem = Problem(domains)
        assert np.all(MinValueHeuristic().choose(problem, 1) == np.array([[0, 0], [1, 2]]))
