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
import numpy as np
import pytest

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.choicepoints import CHOICEPOINT_WIDTH
from nucs.solvers.state import TIGHTENING_UTRAIL_ENTRY_NB
from nucs.solvers.tree_search_solver import STEP_TIGHTENING_NB, allocate_search_arrays, double_rows


class TestTreeSearchSolver:
    def test_allocate_search_arrays_lays_out_the_state(self) -> None:
        """The domains and the entailment flags are views on the state, before the propagator state and the count."""
        problem = Problem([(0, 7), (0, 7), (0, 7)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(3))
        problem.init()
        state, domains, entailed, utrail_headroom, _, utrail_top, utrail_indices, _, choicepoint_top = (
            allocate_search_arrays(problem)
        )
        assert len(state) == 2 * 3 + 1 + problem.state_width + 1
        domains[1, 1] = 5  # (variable 1, its max) is the cell (1 << 1) | 1
        assert state[3] == 5
        entailed[0] = 1  # the flag of propagator 0 follows the 2 * domain_nb bounds
        assert state[6] == 1
        assert len(utrail_indices) == len(state)
        assert utrail_headroom == (
            2 * 3 + 1 + problem.state_trailed_width + 1 + STEP_TIGHTENING_NB * TIGHTENING_UTRAIL_ENTRY_NB
        )
        assert utrail_top.tolist() == [0]
        assert choicepoint_top.tolist() == [1]

    @pytest.mark.parametrize(
        "domain_nb,utrail_size,stack_height",
        [
            # a small problem gets the measured floors
            (2, 1 << 16, 1 << 13),
            # a wide problem gets the sizes derived from it: 16 x headroom and 4 x domain_nb
            (3000, 16 * (2 * 3000 + 1 + STEP_TIGHTENING_NB * TIGHTENING_UTRAIL_ENTRY_NB), 4 * 3000),
        ],
    )
    def test_allocate_search_arrays_sizes_the_undo_trail_and_the_stack(
        self, domain_nb: int, utrail_size: int, stack_height: int
    ) -> None:
        problem = Problem([(0, 1)] * domain_nb)
        problem.init()
        _, _, _, _, utrail_log, _, _, choicepoint_stk, _ = allocate_search_arrays(problem)
        assert utrail_log.shape == (utrail_size, 2)
        assert choicepoint_stk.shape == (stack_height, CHOICEPOINT_WIDTH)

    @pytest.mark.parametrize("width", [2, CHOICEPOINT_WIDTH])  # a row of the undo trail, a row of the stack
    def test_double_rows_keeps_the_rows(self, width: int) -> None:
        array = np.arange(3 * width, dtype=np.int32).reshape(3, width)
        grown = double_rows(array)
        assert grown.shape == (6, width)
        assert grown.dtype == np.int32
        assert grown[:3].tolist() == array.tolist()
