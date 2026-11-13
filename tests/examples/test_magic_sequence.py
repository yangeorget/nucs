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
from collections.abc import Callable

import pytest

from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search
from nucs.statistics import STATS_IDX_SOLUTION_NB, STATS_IDX_SOLVER_CHOICE_NB
from tests.marks import SLOW_WITHOUT_JIT


class TestMagicSequence:
    @pytest.mark.parametrize("size,zero_nb", [(50, 46), (100, 96), pytest.param(200, 196, marks=SLOW_WITHOUT_JIT)])
    def test_find_all(self, size: int, zero_nb: int) -> None:
        problem = MagicSequenceProblem(size)
        solver = BacktrackSolver(problem, searches=[Search(decision_variables=range(size - 1, -1, -1))])
        solutions = solver.find_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
        assert solutions[0][0] == zero_nb

    def test_main(self, run_example_main: Callable[..., BacktrackSolver]) -> None:
        """
        The example splits the domains and tries the upper half first: 19 choices, where min_value makes 111.

        count_eq has x[i] in its list and as its count. Before the write-back of copies (update_domains_of_copies),
        the copies were not intersected, and the search made 20 choices.
        """
        solver = run_example_main("nucs.examples.magic_sequence", "-n", "100")
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == 19
