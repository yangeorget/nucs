from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import Problem


class TestMinValueHeuristic:
    def test_choose(self) -> None:
        shr_domains = np.array([[0, 0], [0, 2]])
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        choice_points: List[NDArray] = []
        MinValueHeuristic().choose(choice_points, problem, 1)
        assert np.all(choice_points == np.array([[0, 0], [1, 2]]))
