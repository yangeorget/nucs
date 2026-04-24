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
from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.car_sequencing.car_sequencing_problem import CarSequencingProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestCarSequencing:
    def test_car_sequencing(self) -> None:
        problem = CarSequencingProblem(
            car_nb=10,
            option_nb=5,
            class_nb=6,
            max_per_block=[1, 2, 1, 2, 1],
            block_size=[2, 3, 3, 5, 5],
            demands=[1, 1, 2, 2, 2, 2],
            requires=[
                [1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ],
        )
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 28
