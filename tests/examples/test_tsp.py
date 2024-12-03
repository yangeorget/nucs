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
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestTSP:
    def test_tsp_1(self) -> None:
        problem = TSPProblem([[0, 2, 1, 2], [2, 0, 2, 1], [1, 2, 0, 2], [2, 1, 2, 0]])
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.shr_domain_nb - 1)
        assert solution is not None
        assert solution[:4].tolist() == [1, 3, 0, 2]
        assert solution[problem.shr_domain_nb - 1] == 6
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 2

    # def test_tsp_gr17(self) -> None:
    #     problem = TSPProblem(GR17)
    #     solver = BacktrackSolver(
    #         problem, var_heuristic_idx=VAR_HEURISTIC_MAX_REGRET, dom_heuristic_idx=DOM_HEURISTIC_MIN_COST
    #     )
    #     solution = solver.minimize(problem.variable_nb - 1)
    #     assert solution is not None
    #     assert solution[problem.variable_nb - 1] == 2085

    # def test_tsp_gr21(self) -> None:
    #     problem = TSPProblem(GR21)
    #     solver = BacktrackSolver(problem)
    #     solution = solver.minimize(problem.variable_nb - 1)
    #     assert solution[problem.variable_nb - 1] == 2707
    #
    # def test_tsp_gr24(self) -> None:
    #     problem = TSPProblem(GR24)
    #     solver = BacktrackSolver(problem)
    #     solution = solver.minimize(problem.variable_nb - 1)
    #     assert solution[problem.variable_nb - 1] == 1272
