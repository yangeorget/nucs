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

from nucs.examples.square.square_problem import SquarePlacementProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


def _assert_valid_packing(problem: SquarePlacementProblem, solution) -> None:  # type: ignore[no-untyped-def]
    """Checks that every square is inside the master rectangle and that no two squares overlap."""
    rects = [
        (int(solution[problem.x(i)]), int(solution[problem.y(i)]), problem.sizes[i]) for i in range(problem.square_nb)
    ]
    for x, y, s in rects:
        assert 0 <= x and x + s <= problem.width, (x, s, problem.width)
        assert 0 <= y and y + s <= problem.height, (y, s, problem.height)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            xi, yi, si = rects[i]
            xj, yj, sj = rects[j]
            assert xi + si <= xj or xj + sj <= xi or yi + si <= yj or yj + sj <= yi, (rects[i], rects[j])


class TestSquare:
    @pytest.mark.parametrize(
        "name",
        [
            "rect_09",  # order-9 simple perfect squared rectangle (33 x 32)
            "square_21",  # order-21 simple perfect squared square, side 112 (the lowest-order SPSS)
        ],
    )
    def test_square_placement(self, name: str) -> None:
        with open(f"datasets/examples/square/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        # the instances are perfect packings: the squares exactly tile the master rectangle
        assert sum(s * s for s in dataset["squares"]) == dataset["width"] * dataset["height"]
        problem = SquarePlacementProblem(dataset["width"], dataset["height"], dataset["squares"])
        solver = BacktrackSolver(problem, searches=problem.recommended_searches(), log_level="ERROR")
        solution = next(solver.solve(), None)
        assert solution is not None
        _assert_valid_packing(problem, solution)
