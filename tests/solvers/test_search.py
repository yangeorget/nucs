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
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
)
from nucs.solvers.search import Search, flatten_searches


class TestSearch:
    def test_flatten_searches(self) -> None:
        """Each search is one part of every flat array, and a variable belongs to the first search that lists it."""
        searches = [
            Search([2, 0], VAR_HEURISTIC_SMALLEST_DOMAIN, [[1, 2]], DOM_HEURISTIC_MIN_VALUE, [[]]),
            Search([0, 1], VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, [[]], DOM_HEURISTIC_MAX_VALUE, [[3], [4]]),
        ]
        flat = flatten_searches(searches, 4)
        assert flat.decision_variables.tolist() == [2, 0, 0, 1]
        assert flat.decision_variables_offsets.tolist() == [0, 2, 4]
        assert flat.decision_variables_per_search() == [[2, 0], [0, 1]]
        assert flat.var_heuristics == [VAR_HEURISTIC_SMALLEST_DOMAIN, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED]
        assert flat.var_heuristic_params.tolist() == [1, 2]
        assert flat.var_heuristic_params_offsets.tolist() == [0, 2, 2]
        assert flat.var_heuristic_params_shapes.tolist() == [[1, 2], [1, 0]]
        assert flat.dom_heuristics == [DOM_HEURISTIC_MIN_VALUE, DOM_HEURISTIC_MAX_VALUE]
        assert flat.dom_heuristic_params.tolist() == [3, 4]
        assert flat.dom_heuristic_params_offsets.tolist() == [0, 0, 2]
        assert flat.dom_heuristic_params_shapes.tolist() == [[1, 0], [2, 1]]
        # 0 is listed by both searches and belongs to the first; 3 is listed by none
        assert flat.variable_searches.tolist() == [0, 1, 0, -1]

    def test_flatten_searches_without_decision_variables_branches_on_all(self) -> None:
        flat = flatten_searches([Search()], 3)
        assert flat.decision_variables.tolist() == [0, 1, 2]
        assert flat.decision_variables_offsets.tolist() == [0, 3]
        assert flat.variable_searches.tolist() == [0, 0, 0]
