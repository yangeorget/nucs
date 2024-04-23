from typing import Iterator

from numpy.typing import NDArray

from ncs.problem import Problem


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem

    def solve(self) -> Iterator[NDArray]:  # type: ignore
        pass
