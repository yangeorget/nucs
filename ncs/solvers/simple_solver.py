import numpy as np

from ncs.solvers.solver import Solver


class SimpleSolver(Solver):
    def solve(self) -> None:
        changes = True
        while changes:
            changes = False
            for constraint in self.problem.constraints:
                constraint_changes = constraint.filter()  # type: ignore
                print(constraint_changes)
                changes |= np.any(constraint_changes)
