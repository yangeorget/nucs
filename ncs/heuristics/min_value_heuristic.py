from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problems.problem import MAX, MIN, Problem


class MinValueHeuristic(ValueHeuristic):
    """
    Chooses the first value of the domain of the variable.
    """

    def choose(self, changes: NDArray, choice_points: List[NDArray], problem: Problem, var_idx: int) -> None:
        domain_idx = problem.dom_indices[var_idx]
        domains = np.copy(problem.shr_domains)
        domains[domain_idx, MIN] += 1
        choice_points.append(domains)
        problem.shr_domains[domain_idx, MAX] = problem.shr_domains[domain_idx, MIN]
        changes[var_idx, MAX] = True
