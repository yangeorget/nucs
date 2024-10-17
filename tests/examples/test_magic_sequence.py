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
# Copyright 2024 - Yan Georget
###############################################################################
import pytest

from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import last_not_instantiated_var_heuristic, min_value_dom_heuristic
from nucs.statistics import STATS_IDX_SOLVER_SOLUTION_NB


class TestMagicSequence:

    @pytest.mark.parametrize("size,zero_nb", [(50, 46), (100, 96), (200, 196)])
    def test_magic_sequence(self, size: int, zero_nb: int) -> None:
        problem = MagicSequenceProblem(size)
        solver = BacktrackSolver(
            problem, var_heuristic=last_not_instantiated_var_heuristic, dom_heuristic=min_value_dom_heuristic
        )
        solutions = solver.find_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1
        assert solutions[0][0] == zero_nb
