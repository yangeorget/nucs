import argparse

import pytest

from ncs.problems.golomb_problem import GolombProblem, index, init_domains
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import MIN, stats_print


class TestGolomb:

    @pytest.mark.parametrize(
        "mark_nb,i,j,idx", [(4, 0, 1, 0), (4, 0, 2, 1), (4, 0, 3, 2), (4, 1, 2, 3), (4, 1, 3, 4), (4, 2, 3, 5)]
    )
    def test_index(self, mark_nb: int, i: int, j: int, idx: int) -> None:
        assert index(mark_nb, i, j) == idx

    def test_init_domains(self) -> None:
        domains = init_domains(6, 4)
        assert domains[:, MIN].tolist() == [1, 3, 6, 1, 3, 1]

    def test_golomb_4_filter(self) -> None:
        problem = GolombProblem(4)
        problem.shared_domains[0] = 1
        problem.shared_domains[1] = 4
        problem.shared_domains[2] = 6
        problem.shared_domains[3] = 3
        problem.shared_domains[4] = 5
        problem.shared_domains[5] = 2
        assert problem.filter()

    @pytest.mark.parametrize("mark_nb,solution_nb", [(4, 6), (5, 11), (6, 17), (7, 25)])
    def test_golomb_4(self, mark_nb: int, solution_nb: int) -> None:
        problem = GolombProblem(mark_nb)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.length)
        assert solution
        assert solution[problem.length] == solution_nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = GolombProblem(args.n)
    solver = BacktrackSolver(problem)
    solution = solver.minimize(problem.length)
    stats_print(solver.statistics)
    print(solution)
    print(solution[problem.length])  # type: ignore
