from typing import List, Optional

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT


class LatinSquareProblem(Problem):
    """
    A simple model for latin squares.
    """

    def __init__(self, symbols: List[int], givens: Optional[List[List[int]]] = None):
        self.symbols = symbols
        self.n = len(symbols)
        if givens is None:
            shr_domains = [(symbols[0], symbols[-1])] * self.n**2
        else:
            shr_domains = [
                (symbols[0], symbols[-1]) if given not in self.symbols else (given, given)
                for line in givens
                for given in line
            ]
        super().__init__(shr_domains)
        for i in range(self.n):
            self.add_propagator((list(range(0 + i * self.n, self.n + i * self.n)), ALG_ALLDIFFERENT, []))
            self.add_propagator((list(range(0 + i, self.n**2 + i, self.n)), ALG_ALLDIFFERENT, []))

    def pretty_print_solution(self, solution: List[int]) -> None:
        for i in range(0, self.n**2, self.n):
            print(solution[i : i + self.n])
