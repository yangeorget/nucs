import argparse
from pprint import pprint

import pytest

from nucs.constants import MIN
from nucs.examples.golomb_problem import GolombProblem, golomb_consistency_algorithm, index, init_domains
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics


class TestGolomb:

    @pytest.mark.parametrize(
        "mark_nb,i,j,idx", [(4, 0, 1, 0), (4, 0, 2, 1), (4, 0, 3, 2), (4, 1, 2, 3), (4, 1, 3, 4), (4, 2, 3, 5)]
    )
    def test_index(self, mark_nb: int, i: int, j: int, idx: int) -> None:
        assert index(mark_nb, i, j) == idx

    def test_init_domains(self) -> None:
        domains = init_domains(6, 4)
        assert domains[:, MIN].tolist() == [1, 3, 6, 1, 3, 1]

    @pytest.mark.parametrize("mark_nb,solution_nb", [(4, 6), (5, 11), (6, 17), (7, 25), (8, 34), (9, 44)])
    def test_golomb(self, mark_nb: int, solution_nb: int) -> None:
        problem = GolombProblem(mark_nb)
        solver = BacktrackSolver(problem, consistency_algorithm=golomb_consistency_algorithm)
        solution = solver.minimize(problem.length_idx)
        assert solution
        assert solution[problem.length_idx] == solution_nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = GolombProblem(args.n)
    solver = BacktrackSolver(problem, consistency_algorithm=golomb_consistency_algorithm)
    solution = solver.minimize(problem.length_idx)
    pprint(get_statistics(solver.statistics))
    print(solution)
    print(solution[problem.length_idx])  # type: ignore
