from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.heuristics.smallest_domain_variable_heuristic import (
    SmallestDomainVariableHeuristic,
)
from ncs.problems.problem import MAX, Problem


class TestSmallestDomainVariableHeuristic:
    def test_choose_variable(self) -> None:
        shr_domains = [(0, 4), (0, 0), (0, 2)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert SmallestDomainVariableHeuristic(MinValueHeuristic()).choose_variable(problem) == 2

    def test_choose(self) -> None:
        shr_domains = [(0, 4), (0, 0), (0, 2)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        choice_points: List[NDArray] = []
        changes = SmallestDomainVariableHeuristic(MinValueHeuristic()).choose(choice_points, problem)
        assert len(choice_points) == 1
        assert np.all(choice_points[0] == np.array([[0, 4], [0, 0], [1, 2]]))
        assert changes is not None
        assert changes[2, MAX]
