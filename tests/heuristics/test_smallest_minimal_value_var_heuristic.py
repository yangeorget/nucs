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
from nucs.heuristics.smallest_minimal_value_var_heuristic import smallest_minimal_value_var_heuristic
from tests.heuristics.var_heuristic_test import call_var_heuristic


class TestSmallestMinimalValueVarHeuristic:
    def test_selects_smallest_minimal_value(self) -> None:
        assert (
            call_var_heuristic(smallest_minimal_value_var_heuristic, [(5, 9), (1, 4), (2, 3)], [0, 1, 2]) == 1
        )  # min 1

    def test_skips_instantiated_variables(self) -> None:
        # variable 0 is bound to 0; even though 0 is the smallest min, a bound variable is never selected
        assert call_var_heuristic(smallest_minimal_value_var_heuristic, [(0, 0), (1, 4)], [0, 1]) == 1

    def test_returns_minus_one_when_all_instantiated(self) -> None:
        assert call_var_heuristic(smallest_minimal_value_var_heuristic, [(3, 3), (7, 7)], [0, 1]) == -1
