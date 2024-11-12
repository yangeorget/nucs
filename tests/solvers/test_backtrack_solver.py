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
from nucs.constants import STATS_LBL_SOLVER_CHOICE_DEPTH, STATS_LBL_SOLVER_SOLUTION_NB
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_RELATION
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestBacktrackSolver:
    def test_solve_and_count(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 10000
        assert statistics[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_solve(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        solver = BacktrackSolver(problem)
        solutions = solver.find_all()
        assert len(solutions) == 4
        assert solutions[0].tolist() == [0, 0]
        assert solutions[1].tolist() == [0, 1]
        assert solutions[2].tolist() == [1, 0]
        assert solutions[3].tolist() == [1, 1]
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 4
        assert statistics[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_solve_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_ALLDIFFERENT, []))
        solver = BacktrackSolver(problem)
        solutions = solver.find_all()
        assert len(solutions) == 6
        assert solutions[0].tolist() == [0, 1, 2]
        assert solutions[1].tolist() == [0, 2, 1]
        assert solutions[2].tolist() == [1, 0, 2]
        assert solutions[3].tolist() == [1, 2, 0]
        assert solutions[4].tolist() == [2, 0, 1]
        assert solutions[5].tolist() == [2, 1, 0]
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 6

    def test_optimize_relation(self) -> None:
        problem = Problem([(-5, 5), (-100, 100)])
        problem.add_propagator(
            ([0, 1], ALG_RELATION, [-5, 25, -4, 16, -3, 9, -2, 4, -1, 1, 0, 0, 1, 1, 2, 4, 3, 9, 4, 16, 5, 25])
        )
        solver = BacktrackSolver(problem)
        solution = solver.minimize(1)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 6
