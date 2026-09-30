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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
from nucs.numpy_helper import flatten_arrays


@dataclass(frozen=True)
class Search:
    """
    One search: the decision variables to branch on, the variable heuristic that picks the next of them, and
    the domain heuristic that reduces it (each with optional parameters). A :class:`BacktrackSolver` runs a
    list of these as a sequential search -- the nested searches are explored in order, each search staying
    active until all of its decision variables are bound. A search is frozen: to change one, make a new one with
    :func:`dataclasses.replace`.
    """

    decision_variables: Iterable[int] | None = None
    var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
    var_heuristic_params: list[list[int]] = field(default_factory=lambda: [[]])
    dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE
    dom_heuristic_params: list[list[int]] = field(default_factory=lambda: [[]])


# the default of BacktrackSolver: one search on every variable with the default heuristics; a frozen tuple of frozen
# searches, so one object serves every solver
DEFAULT_SEARCHES: tuple[Search, ...] = (Search(),)


@dataclass(frozen=True)
class FlatSearches:
    """
    A list of searches as the compiled search reads it: each ragged per-search list is flattened into its
    concatenation and the CSR offsets that delimit the part of each search.
    """

    decision_variables: NDArray
    decision_variables_offsets: NDArray
    var_heuristics: list[int]
    var_heuristic_params: NDArray
    var_heuristic_params_offsets: NDArray
    var_heuristic_params_shapes: NDArray
    dom_heuristics: list[int]
    dom_heuristic_params: NDArray
    dom_heuristic_params_offsets: NDArray
    dom_heuristic_params_shapes: NDArray
    # the search that owns each variable, -1 for a variable no search branches on: last-conflict may only take
    # over the decision of the search that owns the conflict variable
    variable_searches: NDArray

    def decision_variables_per_search(self) -> list[list[int]]:
        """
        Returns the decision variables of each search, as they were before the flattening.

        :return: the decision variables of each search
        :rtype: List[List[int]]
        """
        return [self.decision_variables[a:b].tolist() for a, b in pairwise(self.decision_variables_offsets)]

    def __str__(self) -> str:
        """
        Returns the decision variables and the heuristics of the searches, as the solver logs them.

        :return: the description of the searches
        :rtype: str
        """
        return (
            f"decision domains {self.decision_variables_per_search()}, variable heuristics {self.var_heuristics}"
            f" and domain heuristics {self.dom_heuristics}"
        )


def flatten_searches(searches: Sequence[Search], domain_nb: int) -> FlatSearches:
    """
    Flattens a sequence of searches into the arrays that the compiled search reads.

    A search whose decision variables are None branches on every variable of the problem.

    :param searches: the searches, in the order of the sequential search
    :type searches: Sequence[Search]
    :param domain_nb: the number of variables of the problem
    :type domain_nb: int

    :return: the flattened searches
    :rtype: FlatSearches
    """
    decision_variables_per_search = [
        np.array(
            range(domain_nb) if search.decision_variables is None else list(search.decision_variables),
            dtype=np.uint32,
        )
        for search in searches
    ]
    var_params = [np.array(search.var_heuristic_params, dtype=np.int64) for search in searches]
    dom_params = [np.array(search.dom_heuristic_params, dtype=np.int64) for search in searches]
    decision_variables, decision_variables_offsets = flatten_arrays(decision_variables_per_search)
    var_heuristic_params, var_heuristic_params_offsets = flatten_arrays(var_params)
    dom_heuristic_params, dom_heuristic_params_offsets = flatten_arrays(dom_params)
    variable_searches = np.full(domain_nb, -1, dtype=np.int32)
    for search_idx in reversed(range(len(searches))):  # the first search listing it wins
        variable_searches[decision_variables_per_search[search_idx]] = search_idx
    return FlatSearches(
        decision_variables=decision_variables,
        decision_variables_offsets=decision_variables_offsets,
        var_heuristics=[search.var_heuristic for search in searches],
        var_heuristic_params=var_heuristic_params,
        var_heuristic_params_offsets=var_heuristic_params_offsets,
        var_heuristic_params_shapes=np.array([params.shape for params in var_params], dtype=np.int64),
        dom_heuristics=[search.dom_heuristic for search in searches],
        dom_heuristic_params=dom_heuristic_params,
        dom_heuristic_params_offsets=dom_heuristic_params_offsets,
        dom_heuristic_params_shapes=np.array([params.shape for params in dom_params], dtype=np.int64),
        variable_searches=variable_searches,
    )
