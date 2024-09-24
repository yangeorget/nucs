from pprint import pprint
from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

A, B, D, E, G, L, N, O, R, T = tuple(range(10))


class DonaldProblem(Problem):
    """
    This is the famous crypto-arithmetic puzzle DONALD+GERALD=ROBERT.
    The letters represent digits that have to be all different.
    """

    def __init__(self) -> None:
        super().__init__([(0, 9)] * 10)
        self.add_propagator(
            (
                [A, B, D, E, G, L, N, O, R, T],
                ALG_AFFINE_EQ,
                [200, -1000, 100002, 9900, 100000, 20, 1000, 0, -99010, -1, 0],
            )
        )
        self.add_propagator(
            ([A, B, D, E, G, L, N, O, R, T], ALG_ALLDIFFERENT, []),
        )

    def pretty_print_solution(self, solution: List[int]) -> None:
        print(
            {
                "A": solution[A],
                "B": solution[B],
                "D": solution[D],
                "E": solution[E],
                "G": solution[G],
                "L": solution[L],
                "N": solution[N],
                "O": solution[O],
                "R": solution[R],
                "T": solution[T],
            }
        )


if __name__ == "__main__":
    problem = DonaldProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    for solution in solver.solve():
        problem.pretty_print_solution(solution)
    pprint(get_statistics(solver.statistics))
