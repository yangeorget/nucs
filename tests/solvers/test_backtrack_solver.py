import numpy as np

from ncs.problems.problem import (
    ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ,
    ALGORITHM_DUMMY,
    ALGORITHM_SUM,
    Problem,
)
from ncs.propagators.propagator import Propagator
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import (
    STATS_PROBLEM_FILTERS_NB,
    STATS_SOLVER_BACKTRACKS_NB,
    STATS_SOLVER_CP_MAX,
    STATS_SOLVER_SOLUTIONS_NB,
)


class TestBacktrackSolver:
    def test_solve_and_count(self) -> None:
        shr_domains = [(0, 99), (0, 99)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([], ALGORITHM_DUMMY)])
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 10000
        assert solver.statistics[STATS_SOLVER_CP_MAX] == 2

    def test_solve(self) -> None:
        shr_domains = [(0, 1), (0, 1)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([], ALGORITHM_DUMMY)])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 4
        assert solutions[0] == [0, 0]
        assert solutions[1] == [0, 1]
        assert solutions[2] == [1, 0]
        assert solutions[3] == [1, 1]
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 4
        assert solver.statistics[STATS_SOLVER_CP_MAX] == 2

    def test_solve_sum_1(self) -> None:
        shr_domains = [(0, 2), (0, 2), (4, 6)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([2, 0, 1], ALGORITHM_SUM)])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert solutions == [[2, 2, 4]]
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 1
        assert solver.statistics[STATS_PROBLEM_FILTERS_NB] == 1
        assert solver.statistics[STATS_SOLVER_CP_MAX] == 0
        assert solver.statistics[STATS_SOLVER_BACKTRACKS_NB] == 0

    def test_solve_sum_3(self) -> None:
        shr_domains = [(0, 1), (0, 1), (0, 1)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([2, 0, 1], ALGORITHM_SUM)])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 3
        assert solutions[0] == [0, 0, 0]
        assert solutions[1] == [0, 1, 1]
        assert solutions[2] == [1, 0, 1]
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 3
        assert solver.statistics[STATS_SOLVER_CP_MAX] == 2

    def test_solve_sum_ko(self) -> None:
        shr_domains = [(1, 2), (1, 2), (0, 1)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([2, 0, 1], ALGORITHM_SUM)])
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 0
        assert solver.statistics[STATS_PROBLEM_FILTERS_NB] == 1
        assert solver.statistics[STATS_SOLVER_CP_MAX] == 0

    def test_solve_alldifferent(self) -> None:
        shr_domains = [(0, 2), (0, 2), (0, 2)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.set_propagators([([0, 1, 2], ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ)])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 6
        assert solutions[0] == [0, 1, 2]
        assert solutions[1] == [0, 2, 1]
        assert solutions[2] == [1, 0, 2]
        assert solutions[3] == [1, 2, 0]
        assert solutions[4] == [2, 0, 1]
        assert solutions[5] == [2, 1, 0]
