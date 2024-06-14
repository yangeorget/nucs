from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problems.problem import MAX, MIN, Problem


class MinValueHeuristic(ValueHeuristic):
    """
    Chooses the first value of the domain of the variable.
    """

    def choose(self, choice_points: List[NDArray], problem: Problem, var_idx: int) -> Optional[NDArray]:
        domain_idx = problem.dom_indices[var_idx]
        domains = np.copy(problem.shr_domains)
        domains[domain_idx, MIN] += 1
        choice_points.append(domains)
        problem.shr_domains[domain_idx, MAX] = problem.shr_domains[domain_idx, MIN]  # we update THE shared domain
        changes = np.zeros((problem.size, 2), dtype=bool)
        changes[:, MAX] = problem.dom_indices == domain_idx  # we changed the max of all variables sharing that domain
        return changes
