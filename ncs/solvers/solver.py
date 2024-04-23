from typing import Optional

from numpy._typing import NDArray

from ncs.problem import Problem


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem

    def solve(self) -> Optional[NDArray]:  # type: ignore
        pass
