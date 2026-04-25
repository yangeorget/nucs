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
from typing import List

import pytest

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.cryptarithmetic.cryptarithmetic_problem import CryptarithmeticProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestCryptarithmetic:
    @pytest.mark.parametrize(
        "path, values",
        [
            ("datasets/cryptarithmetic/donald.json", [4, 3, 5, 9, 1, 8, 6, 2, 7, 0]),
            ("datasets/cryptarithmetic/sendmore.json", [7, 5, 1, 6, 0, 8, 9, 2]),
        ],
    )
    def test_puzzles(self, path: str, values: List[int]) -> None:
        with open(path, "r") as json_file:
            dataset = json.load(json_file)
            problem = CryptarithmeticProblem(dataset)
            solver = BacktrackSolver(
                problem, var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic=DOM_HEURISTIC_MIN_VALUE
            )
            solutions = solver.find_all()
            assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
            assert solutions[0].tolist() == values
