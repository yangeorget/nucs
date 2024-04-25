import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problem import MAX, MIN, Problem


class MinValueHeuristic(ValueHeuristic):
    def make_value_choice(self, problem: Problem, idx: int) -> NDArray:
        # print("makeVariableChoice()")
        domains = np.copy(problem.domains)
        problem.domains[idx, MAX] = problem.domains[idx, MIN]
        domains[idx, MIN] += 1
        return domains
