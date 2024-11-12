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
from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestBIBD:
    def test_6_10_5_3_2(self) -> None:
        problem = BIBDProblem(6, 10, 5, 3, 2)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1

    def test_7_7_3_3_1(self) -> None:
        problem = BIBDProblem(7, 7, 3, 3, 1)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1

    def test_8_14_7_4_3(self) -> None:
        problem = BIBDProblem(8, 14, 7, 4, 3)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 92
