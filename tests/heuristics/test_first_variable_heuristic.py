from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import MAX, Problem


class TestFirstVariableHeuristic:
    def test_choose_variable(self) -> None:
        shr_domains = [(0, 0), (0, 2)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert FirstVariableHeuristic(MinValueHeuristic()).choose_variable(problem) == 1

    def test_choose(self) -> None:
        shr_domains = [(0, 0), (0, 2)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        choice_points: List[NDArray] = []
        changes = FirstVariableHeuristic(MinValueHeuristic()).choose(choice_points, problem)
        assert len(choice_points) == 1
        assert np.all(choice_points[0] == np.array([[0, 0], [1, 2]]))
        assert changes is not None
        assert changes[1, MAX]
