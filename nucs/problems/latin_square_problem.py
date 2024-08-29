from typing import List, Optional

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT


class LatinSquareProblem(Problem):
    """
    A simple model for latin squares.
    """

    def __init__(self, colors: List[int], givens: Optional[List[List[int]]] = None):
        """
        Inits the latin square.
        :colors: the possible values for the cells,
        usually [0, ..., n-1] except in some cases (Sudokus) where [1, ..., n] is preferred;
        the number of colors is also the size of the square
        :givens: initial values for the cells, any value different of the possible colors is used as a wildcard
        """
        self.colors = colors
        self.n = len(colors)
        if givens is None:
            shr_domains = [(colors[0], colors[-1])] * self.n**2
        else:
            shr_domains = [
                (colors[0], colors[-1]) if given not in self.colors else (given, given)
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


class LatinSquareRCProblem(LatinSquareProblem):
    """
    A full model for latin squares with 3 kinds of variables:
    - n*n variables for the values, these variables are indexed by rows then columns
    - n*n variables for the rows, these variables are indexed by colors then columns
    - n*n variables for the columns, these variables are indexed by colors then rows
    """

    # TODO:
