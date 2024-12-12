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
import pytest

from nucs.constants import OPT_PRUNE, OPT_RESET, STATS_LBL_SOLVER_SOLUTION_NB
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_RELATION
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver


class TestMultiprocessingSolver:
    def test_solve1(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in (problem.split(4, 0))])
        solutions = solver.find_all()
        assert len(solutions) == 10000
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 10000

    def test_solve_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_ALLDIFFERENT, []))
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in (problem.split(3, 0))])
        solutions = solver.find_all()
        assert len(solutions) == 6
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == 6

    @pytest.mark.parametrize(
        "mode, split_var, split_nb, solution_nb",
        [
            (OPT_PRUNE, 0, 1, 6),
            (OPT_RESET, 0, 1, 6),
            (OPT_PRUNE, 0, 2, 6),
            (OPT_RESET, 0, 2, 6),
            (OPT_PRUNE, 0, 3, 6),
            (OPT_RESET, 0, 3, 6),
            (OPT_PRUNE, 1, 1, 6),
            (OPT_RESET, 1, 1, 6),
            (OPT_PRUNE, 1, 2, 6),
            (OPT_RESET, 1, 2, 6),
            (OPT_PRUNE, 1, 3, 6),
            (OPT_RESET, 1, 3, 6),
        ],
    )
    def test_minimize_relation(self, mode: str, split_var: int, split_nb: int, solution_nb: int) -> None:
        problem = Problem([(-5, 0), (-60, 60)])
        problem.add_propagator(([0, 1], ALG_RELATION, [-5, 25, -4, 16, -3, 9, -2, 4, -1, 1, 0, 0]))
        solver = MultiprocessingSolver([BacktrackSolver(prob) for prob in problem.split(split_nb, split_var)])
        solution = solver.minimize(1, mode=mode)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        statistics = solver.get_statistics()
        assert statistics[STATS_LBL_SOLVER_SOLUTION_NB] == solution_nb
