from typing import Optional

import numpy as np
from numpy._typing import NDArray

from ncs.problem import Problem


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem

    def filter(self) -> bool:
        print("filter()")
        changes = True
        while changes:
            changes = False
            for constraint in self.problem.constraints:
                constraint_changes = constraint.filter(self.problem)
                print(f"constraint_changes={constraint_changes}")
                if self.problem.is_inconsistent():
                    return False
                changes |= np.any(constraint_changes)  # type: ignore
        return True

    def solve(self) -> Optional[NDArray]:  # type: ignore
      pass
