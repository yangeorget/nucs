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
from nucs.examples.car_sequencing.car_sequencing_datasets import DATASETS
from nucs.examples.car_sequencing.car_sequencing_problem import CarSequencingProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestCarSequencing:
    def test_car_sequencing(self) -> None:
        problem = CarSequencingProblem(DATASETS["ECAI88"])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 28
