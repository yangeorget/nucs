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
from collections.abc import Iterable
from dataclasses import dataclass, field

from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED


@dataclass
class Search:
    """
    One search: the decision variables to branch on, the variable heuristic that picks the next of them, and
    the domain heuristic that reduces it (each with optional parameters). A :class:`BacktrackSolver` runs a
    list of these as a sequential search -- the nested searches are explored in order, each search staying
    active until all of its decision variables are bound.
    """

    decision_variables: Iterable[int] | None = None
    var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
    var_heuristic_params: list[list[int]] = field(default_factory=lambda: [[]])
    dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE
    dom_heuristic_params: list[list[int]] = field(default_factory=lambda: [[]])
