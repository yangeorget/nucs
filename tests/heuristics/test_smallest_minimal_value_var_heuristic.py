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

from nucs.constants import MAX, MIN
from nucs.heuristics.smallest_minimal_value_var_heuristic import smallest_minimal_value_var_heuristic


def _domains_stk(domains, top=0, height=2):  # type: ignore[no-untyped-def]
    stk = np.zeros((height, len(domains), 2), dtype=np.int32)
    for i, (lo, hi) in enumerate(domains):
        stk[top, i, MIN] = lo
        stk[top, i, MAX] = hi
    return stk


class TestSmallestMinimalValueVarHeuristic:
    def test_selects_smallest_minimal_value(self) -> None:
        stk = _domains_stk([(5, 9), (1, 4), (2, 3)])
        variables = np.array([0, 1, 2], dtype=np.uint32)
        assert smallest_minimal_value_var_heuristic(variables, stk, 0, np.empty((1, 0))) == 1  # min 1

    def test_skips_instantiated_variables(self) -> None:
        # variable 0 is bound to 0; even though 0 is the smallest min, a bound variable is never selected
        stk = _domains_stk([(0, 0), (1, 4)])
        variables = np.array([0, 1], dtype=np.uint32)
        assert smallest_minimal_value_var_heuristic(variables, stk, 0, np.empty((1, 0))) == 1

    def test_returns_minus_one_when_all_instantiated(self) -> None:
        stk = _domains_stk([(3, 3), (7, 7)])
        variables = np.array([0, 1], dtype=np.uint32)
        assert smallest_minimal_value_var_heuristic(variables, stk, 0, np.empty((1, 0))) == -1
