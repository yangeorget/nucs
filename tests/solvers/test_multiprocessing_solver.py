###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_RELATION
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver
from nucs.statistics import (
    STATS_LBL_OPTIMIZER_SOLUTION_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_SOLUTION_NB,
    get_statistics,
)


class TestMultiprocessingSolver:
    def test_solve_and_count(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        problems = problem.split(4, 0)
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
        solver.solve_all()
        assert get_statistics(solver.statistics)[STATS_LBL_SOLVER_SOLUTION_NB] == 10000
        assert get_statistics(solver.statistics)[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_solve(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        problems = problem.split(2, 0)
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
        solutions = solver.find_all()
        assert len(solutions) == 4
        assert get_statistics(solver.statistics)[STATS_LBL_SOLVER_SOLUTION_NB] == 4

    def test_solve_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_ALLDIFFERENT, []))
        problems = problem.split(3, 0)
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
        solutions = solver.find_all()
        assert len(solutions) == 6
        assert get_statistics(solver.statistics)[STATS_LBL_SOLVER_SOLUTION_NB] == 6

    def test_minimize_relation(self) -> None:
        problem = Problem([(-5, 5), (-100, 100)])
        problem.add_propagator(
            ([0, 1], ALG_RELATION, [-5, 25, -4, 16, -3, 9, -2, 4, -1, 1, 0, 0, 1, 1, 2, 4, 3, 9, 4, 16, 5, 25])
        )
        problems = problem.split(5, 0)
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
        solution = solver.minimize(1)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        assert get_statistics(solver.statistics)[STATS_LBL_OPTIMIZER_SOLUTION_NB] >= 6
