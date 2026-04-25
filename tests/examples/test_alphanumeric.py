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
from nucs.examples.alphanumeric.alphanumeric_problem import AlphanumericProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestAlphanumeric:
    @pytest.mark.parametrize(
        "path, values",
        [
            (
                "datasets/alphanumeric/alpha.json",
                [
                    5,
                    13,
                    9,
                    16,
                    20,
                    4,
                    24,
                    21,
                    25,
                    17,
                    23,
                    2,
                    8,
                    12,
                    10,
                    19,
                    7,
                    11,
                    15,
                    3,
                    1,
                    26,
                    6,
                    22,
                    14,
                    18,
                ],
            )
        ],
    )
    def test_puzzles(self, path: str, values: List[int]) -> None:
        with open(path, "r") as json_file:
            dataset = json.load(json_file)
            problem = AlphanumericProblem(dataset)
            solver = BacktrackSolver(problem, var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN)
            solutions = solver.find_all()
            assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
            assert solutions[0].tolist() == values
