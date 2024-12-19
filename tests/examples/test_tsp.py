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

from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.examples.tsp.tsp_instances import TSP_INSTANCES
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.examples.tsp.tsp_var_heuristic import tsp_var_heuristic
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_COST, register_var_heuristic
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_SHAVING


class TestTSP:
    def test_tsp_1(self) -> None:
        problem = TSPProblem([[0, 2, 1, 2], [2, 0, 2, 1], [1, 2, 0, 2], [2, 1, 2, 0]])
        solver = BacktrackSolver(problem, decision_domains=[0, 1, 2, 3])
        solution = solver.minimize(problem.total_cost)
        assert solution is not None
        assert solution[:4].tolist() == [1, 3, 0, 2]
        assert solution[problem.total_cost] == 6
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 2

    @pytest.mark.parametrize(
        "name, minimum",
        [
            ("GR17", 2085),
            #  ("GR21", 2707),
            #  ("GR24", 1272)
        ],
    )
    def test_tsp_gr(self, name: str, minimum: int) -> None:
        tsp_instance = TSP_INSTANCES[name]
        n = len(tsp_instance)
        problem = TSPProblem(tsp_instance)
        tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
        costs = tsp_instance + tsp_instance
        solver = BacktrackSolver(
            problem,
            consistency_alg_idx=CONSISTENCY_ALG_SHAVING,
            decision_domains=list(range(0, 2 * n)),
            var_heuristic_idx=tsp_var_heuristic_idx,
            var_heuristic_params=costs,
            dom_heuristic_idx=DOM_HEURISTIC_MIN_COST,
            dom_heuristic_params=costs,
        )
        solution = solver.minimize(problem.total_cost)
        assert solution is not None
        assert solution[problem.total_cost] == minimum
