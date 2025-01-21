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
import time
from multiprocessing import Queue
from typing import Optional

import enlighten
import numpy as np
import pytest
from numpy.typing import NDArray

from nucs.constants import OPTIM_PRUNE, OPTIM_RESET, STATS_LBL_SOLUTION_NB, STATS_MAX
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_RELATION
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver
from nucs.solvers.queue_solver import QueueSolver


class TestMultiprocessingSolver:
    def test_find_one(self) -> None:
        class FastSolver(QueueSolver):
            def get_progress_bar(self, manager: enlighten.Manager) -> Optional[enlighten.Counter]:
                return None

            def get_statistics_as_array(self) -> NDArray:
                return np.array(0)

            def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
                solution_queue.put((processor_idx, np.array(0), np.array([0] * STATS_MAX, dtype=np.int64)))

            def minimize_and_queue(
                self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str
            ) -> None:
                pass

            def maximize_and_queue(
                self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str
            ) -> None:
                pass

        class SlowSolver(QueueSolver):
            def get_progress_bar(self, manager: enlighten.Manager) -> Optional[enlighten.Counter]:
                return None

            def get_statistics_as_array(self) -> NDArray:
                return np.array(0)

            def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
                time.sleep(10)
                solution_queue.put((processor_idx, None, np.array([0] * STATS_MAX, dtype=np.int64)))

            def minimize_and_queue(
                self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str
            ) -> None:
                pass

            def maximize_and_queue(
                self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str
            ) -> None:
                pass

        solver = MultiprocessingSolver([SlowSolver(), FastSolver()])
        assert solver.find_one() is not None

    def test_find_all_1(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in (problem.split(4, 0))])
        solutions = solver.find_all()
        assert len(solutions) == 10000
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 10000

    def test_find_all_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_ALLDIFFERENT, []))
        solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in (problem.split(3, 0))])
        solutions = solver.find_all()
        assert len(solutions) == 6
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 6

    @pytest.mark.parametrize(
        "mode, split_var, split_nb, solution_nb",
        [
            (OPTIM_PRUNE, 0, 1, 6),
            (OPTIM_RESET, 0, 1, 6),
            (OPTIM_PRUNE, 0, 2, 6),
            (OPTIM_RESET, 0, 2, 6),
            (OPTIM_PRUNE, 0, 3, 6),
            (OPTIM_RESET, 0, 3, 6),
            (OPTIM_PRUNE, 1, 1, 6),
            (OPTIM_RESET, 1, 1, 6),
            (OPTIM_PRUNE, 1, 2, 6),
            (OPTIM_RESET, 1, 2, 6),
            (OPTIM_PRUNE, 1, 3, 6),
            (OPTIM_RESET, 1, 3, 6),
        ],
    )
    def test_minimize_relation(self, mode: str, split_var: int, split_nb: int, solution_nb: int) -> None:
        problem = Problem([(-5, 0), (-60, 60)])
        problem.add_propagator(([0, 1], ALG_RELATION, [-5, 25, -4, 16, -3, 9, -2, 4, -1, 1, 0, 0]))
        solver = MultiprocessingSolver([BacktrackSolver(prob) for prob in problem.split(split_nb, split_var)])
        solution = solver.minimize(1, mode=mode)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == solution_nb
