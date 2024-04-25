import numpy as np
from numpy._typing import NDArray

from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problem import MAX, MIN, Problem


class MinVariableHeuristic(VariableHeuristic):
    def makeVariableChoice(self, problem: Problem, idx: int) -> NDArray:
        # print("makeVariableChoice()")
        domains = np.copy(problem.domains)
        problem.domains[idx, MAX] = problem.domains[idx, MIN]
        domains[idx, MIN] += 1
        return domains
