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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import List

import pytest

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestAllIntervalSeries:
    @pytest.mark.parametrize(
        "size,values",
        [
            (8, [0, 7, 1, 6, 2, 5, 3, 4]),
            (9, [0, 8, 1, 7, 2, 6, 3, 5, 4]),
        ],
    )
    def test_find_one(self, size: int, values: List[int]) -> None:
        problem = AllIntervalSeriesProblem(size, True)
        solver = BacktrackSolver(problem)
        solution = solver.find_one()
        assert solution is not None
        assert solution[:size].tolist() == values

    @pytest.mark.parametrize(
        "size,solution_nb",
        [
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 6),
            (7, 8),
            (8, 10),
            (9, 30),
        ],
    )
    def test_solve_all(self, size: int, solution_nb: int) -> None:
        problem = AllIntervalSeriesProblem(size, True)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
