import argparse
from pprint import pprint

from nucs.problems.latin_square_problem import M_COLOR, M_COLUMN, M_ROW, LatinSquareRCProblem
from nucs.propagators.propagators import ALG_ELEMENT_LIV
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics


class QuasigroupProblem(LatinSquareRCProblem):
    """
    CSPLIB problem #3 - https://www.csplib.org/Problems/prob003/
    """

    def __init__(self, n: int):
        super().__init__(n)
        # idempotence
        for model in [M_COLOR, M_ROW, M_COLUMN]:
            for i in range(n):
                self.shr_domains_lst[self.cell(i, i, model)] = i
        # symmetry breaking
        for i in range(1, n):
            self.shr_domains_lst[self.cell(i, n - 1)] = (i - 1, n - 1)


class Quasigroup5Problem(QuasigroupProblem):
    """
    Defined by:
       ((b∗a)∗b)∗b=a
    Can be enforced by:
       color[color[j, i] ,j] = row[i, j]
    which avoids the creation of additional variables
    """

    def __init__(self, n: int):
        super().__init__(n)
        for j in range(n):
            for i in range(n):
                if i != j:
                    self.add_propagator(
                        (
                            [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), self.cell(i, j, M_ROW)],
                            ALG_ELEMENT_LIV,
                            [],
                        ),
                        0,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = Quasigroup5Problem(args.n)
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solver.solve_all()
    pprint(get_statistics(solver.statistics))