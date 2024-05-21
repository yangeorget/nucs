from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.problem import Problem


class TestFirstVariableHeuristic:
    def test_make_variable_choice(self) -> None:
        domains = np.array([[0, 0], [0, 2]])
        problem = Problem(domains)
        assert FirstVariableHeuristic().make_variable_choice(problem) == 1

    def test_make_choice(self) -> None:
        domains = np.array([[0, 0], [0, 2]])
        problem = Problem(domains)
        choice_points: List[NDArray] = []
        assert FirstVariableHeuristic().make_choice(choice_points, problem)
        assert len(choice_points) == 1
        assert np.all(choice_points[0] == np.array([[0, 0], [1, 2]]))
