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
import pytest

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.choice_points import CHOICE_POINT_WIDTH
from nucs.solvers.search_arrays import STEP_TIGHTENING_NB, TIGHTENING_TRAIL_ENTRY_NB, allocate_search_arrays


class TestSearchArrays:
    def test_allocate_search_arrays_lays_out_the_state(self) -> None:
        """The domains and the entailment flags are views on the state, before the propagator state and the count."""
        problem = Problem([(0, 7), (0, 7), (0, 7)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(3))
        problem.init()
        state, domains, entailed, trail_headroom, _, trail_top, trail_indices, _, choice_point_top = (
            allocate_search_arrays(problem)
        )
        assert len(state) == 2 * 3 + 1 + problem.state_width + 1
        domains[1, 1] = 5  # (variable 1, its max) is the cell (1 << 1) | 1
        assert state[3] == 5
        entailed[0] = 1  # the flag of propagator 0 follows the 2 * domain_nb bounds
        assert state[6] == 1
        assert len(trail_indices) == len(state)
        assert trail_headroom == (
            2 * 3 + 1 + problem.state_trailed_width + 1 + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB
        )
        assert trail_top.tolist() == [0]
        assert choice_point_top.tolist() == [1]

    @pytest.mark.parametrize(
        "domain_nb,trail_size,stack_height",
        [
            # a small problem gets the measured floors
            (2, 1 << 16, 1 << 13),
            # a wide problem gets the sizes derived from it: 16 x headroom and 4 x domain_nb
            (3000, 16 * (2 * 3000 + 1 + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB), 4 * 3000),
        ],
    )
    def test_allocate_search_arrays_sizes_the_trail_and_the_stack(
        self, domain_nb: int, trail_size: int, stack_height: int
    ) -> None:
        problem = Problem([(0, 1)] * domain_nb)
        problem.init()
        _, _, _, _, trail_log, _, _, choice_point_stk, _ = allocate_search_arrays(problem)
        assert trail_log.shape == (trail_size, 2)
        assert choice_point_stk.shape == (stack_height, CHOICE_POINT_WIDTH)
