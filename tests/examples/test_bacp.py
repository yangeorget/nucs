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

from nucs.examples.bacp.bacp_problem import BACPProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestBACP:
    def test_balanced(self) -> None:
        dataset = {
            "n_courses": 3,
            "n_periods": 2,
            "load_per_period_lb": 0,
            "load_per_period_ub": 10,
            "courses_per_period_lb": 1,
            "courses_per_period_ub": 2,
            "course_load": [1, 2, 3],
            "prerequisites": [],
        }
        problem = BACPProblem(dataset)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.max_load)
        assert solution is not None
        assert solution[problem.max_load] == 3
        loads = [int(solution[problem.load(j)]) for j in range(problem.n_periods)]
        assert sorted(loads) == [3, 3]

    def test_infeasible_load_bounds(self) -> None:
        # Total load is 6 but each period must have load <= 2, and we only have 2 periods (capacity 4).
        dataset = {
            "n_courses": 3,
            "n_periods": 2,
            "load_per_period_lb": 0,
            "load_per_period_ub": 2,
            "courses_per_period_lb": 1,
            "courses_per_period_ub": 3,
            "course_load": [1, 2, 3],
            "prerequisites": [],
        }
        problem = BACPProblem(dataset)
        solver = BacktrackSolver(problem)
        assert solver.minimize(problem.max_load) is None

    @pytest.mark.parametrize(
        "path, max_load",
        [
            # ("datasets/examples/bacp/bacp-1.json", 28), slow
            # ("datasets/examples/bacp/bacp-2.json", ?), ?
            ("datasets/examples/bacp/bacp-3.json", 30),
            ("datasets/examples/bacp/bacp-4.json", 44),
            ("datasets/examples/bacp/bacp-5.json", 26),
            ("datasets/examples/bacp/bacp-6.json", 26),
            ("datasets/examples/bacp/bacp-7.json", 27),
            ("datasets/examples/bacp/bacp-8.json", 30),
            ("datasets/examples/bacp/bacp-9.json", 38),
        ],
    )
    def test_datasets(self, path: str, max_load: int) -> None:
        with open(path, "r") as json_file:
            dataset = json.load(json_file)
        problem = BACPProblem(dataset)
        solver = BacktrackSolver(
            problem, decision_variables=range(dataset["n_courses"]), var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN
        )
        solution = solver.minimize(problem.max_load)
        assert solution is not None
        assert solution[problem.max_load] == max_load
