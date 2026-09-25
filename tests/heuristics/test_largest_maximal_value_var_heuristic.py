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
from nucs.heuristics.largest_maximal_value_var_heuristic import largest_maximal_value_var_heuristic
from tests.heuristics.var_heuristic_test import call_var_heuristic


class TestLargestMaximalValueVarHeuristic:
    def test_selects_largest_maximal_value(self) -> None:
        assert (
            call_var_heuristic(largest_maximal_value_var_heuristic, [(5, 9), (1, 4), (2, 3)], [0, 1, 2]) == 0
        )  # max 9

    def test_skips_instantiated_variables(self) -> None:
        # variable 0 is bound to 5; even though 5 is the largest max, a bound variable is never selected
        assert call_var_heuristic(largest_maximal_value_var_heuristic, [(5, 5), (1, 4)], [0, 1]) == 1

    def test_returns_minus_one_when_all_instantiated(self) -> None:
        assert call_var_heuristic(largest_maximal_value_var_heuristic, [(3, 3), (7, 7)], [0, 1]) == -1
