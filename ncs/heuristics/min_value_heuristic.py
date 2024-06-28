from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problems.problem import Problem
from ncs.utils import MAX, MIN


class MinValueHeuristic(ValueHeuristic):
    """
    Chooses the first value of the domain of the variable.
    """

    def choose(self, choice_points: List[NDArray], problem: Problem, var_idx: int) -> Optional[NDArray]:
        domain_idx = problem.dom_indices[var_idx]
        domains = np.copy(problem.shr_domains)
        domains[domain_idx, MIN] += 1
        choice_points.append(domains)
        problem.shr_domains[domain_idx, MAX] = problem.shr_domains[domain_idx, MIN]
        shr_changes = np.zeros((len(problem.shr_domains), 2), dtype=np.bool)
        shr_changes[domain_idx, MAX] = True
        return shr_changes
