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
# Copyright 2024-2026 - Yan Georget
###############################################################################
import json

import pytest

from nucs.constants import DOMAIN_MIN
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.examples.tsp.tsp_var_heuristic import tsp_var_heuristic
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_COST, register_var_heuristic
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import OPTIM_PRUNE
from nucs.statistics import STATS_IDX_SOLUTION_NB


class TestTSP:
    def test_find_best(self) -> None:
        problem = TSPProblem([[0, 2, 1, 2], [2, 0, 2, 1], [1, 2, 0, 2], [2, 1, 2, 0]])
        solver = BacktrackSolver(problem, decision_variables=[0, 1, 2, 3])
        solution = solver.find_best(problem.total_cost, bound=DOMAIN_MIN)
        assert solution is not None
        assert solution[:4].tolist() == [1, 3, 0, 2]
        assert solution[problem.total_cost] == 6
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 2

    @pytest.mark.parametrize(
        "name, minimum",
        [
            ("gr17", 2085),
            ("gr21", 2707),
            # ("gr24", 1272)
        ],
    )
    def test_find_best_datasets(self, name: str, minimum: int) -> None:
        with open(f"datasets/examples/tsp/{name}.json", "r") as json_file:
            costs = json.load(json_file)["costs"]
            n = len(costs)
            problem = TSPProblem(costs)
            costs = costs + costs
            tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
            solver = BacktrackSolver(
                problem,
                decision_variables=range(2 * n),
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
            )
            solution = solver.find_best(problem.total_cost, bound=DOMAIN_MIN, mode=OPTIM_PRUNE)
            assert solution is not None
            assert solution[problem.total_cost] == minimum
