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
from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search
from nucs.solvers.solver import OPTIM_PRUNE
from nucs.statistics import STATS_IDX_SOLUTION_NB, STATS_IDX_SOLVER_CHOICE_NB

# without the JIT, the full instances take about 100,000 choices: 116 s and 156 s on a fast machine, and much more under
# coverage on the CI
SLOW_WITHOUT_JIT = pytest.mark.skipif(bool(NUMBA_DISABLE_JIT), reason="too slow without the JIT")


class TestTSP:
    def test_find_best(self) -> None:
        problem = TSPProblem([[0, 2, 1, 2], [2, 0, 2, 1], [1, 2, 0, 2], [2, 1, 2, 0]])
        solver = BacktrackSolver(problem, searches=[Search(decision_variables=[0, 1, 2, 3])])
        solution = solver.find_best(problem.total_cost, bound=DOMAIN_MIN)
        assert solution is not None
        assert solution[:4].tolist() == [1, 3, 0, 2]
        assert solution[problem.total_cost] == 6
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 2

    @pytest.mark.parametrize(
        "name, city_nb, minimum, choice_nb",
        [
            # the first 10 cities of gr17, with the same search as the full instances: 153 choices, which also run
            # without the JIT; a brute force over all the tours gives the same optimum
            ("gr17", 10, 1637, 153),
            # the case above runs the same code without the JIT
            pytest.param("gr17", 17, 2085, 94926, marks=SLOW_WITHOUT_JIT),
            pytest.param("gr21", 21, 2707, 108196, marks=SLOW_WITHOUT_JIT),
            # ("gr24", 24, 1272, ...)
        ],
    )
    def test_find_best_datasets(self, name: str, city_nb: int, minimum: int, choice_nb: int) -> None:
        with open(f"datasets/examples/tsp/{name}.json", "r") as json_file:
            costs = [row[:city_nb] for row in json.load(json_file)["costs"][:city_nb]]
        problem = TSPProblem(costs)
        costs = costs + costs
        tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
        solver = BacktrackSolver(
            problem,
            searches=[
                Search(
                    decision_variables=range(2 * city_nb),
                    var_heuristic=tsp_var_heuristic_idx,
                    var_heuristic_params=costs,
                    dom_heuristic=DOM_HEURISTIC_MIN_COST,
                    dom_heuristic_params=costs,
                )
            ],
        )
        solution = solver.find_best(problem.total_cost, bound=DOMAIN_MIN, mode=OPTIM_PRUNE)
        assert solution is not None
        assert solution[problem.total_cost] == minimum
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == choice_nb
