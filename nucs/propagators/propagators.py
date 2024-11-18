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
from typing import Callable

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.propagators.affine_eq_propagator import (
    compute_domains_affine_eq,
    get_complexity_affine_eq,
    get_triggers_affine_eq,
)
from nucs.propagators.affine_geq_propagator import (
    compute_domains_affine_geq,
    get_complexity_affine_geq,
    get_triggers_affine_geq,
)
from nucs.propagators.affine_leq_propagator import (
    compute_domains_affine_leq,
    get_complexity_affine_leq,
    get_triggers_affine_leq,
)
from nucs.propagators.alldifferent_propagator import (
    compute_domains_alldifferent,
    get_complexity_alldifferent,
    get_triggers_alldifferent,
)
from nucs.propagators.and_propagator import compute_domains_and, get_complexity_and, get_triggers_and
from nucs.propagators.count_eq_propagator import (
    compute_domains_count_eq,
    get_complexity_count_eq,
    get_triggers_count_eq,
)
from nucs.propagators.dummy_propagator import compute_domains_dummy, get_complexity_dummy, get_triggers_dummy
from nucs.propagators.element_lic_propagator import (
    compute_domains_element_lic,
    get_complexity_element_lic,
    get_triggers_element_lic,
)
from nucs.propagators.element_liv_propagator import (
    compute_domains_element_liv,
    get_complexity_element_liv,
    get_triggers_element_liv,
)
from nucs.propagators.exactly_eq_propagator import (
    compute_domains_exactly_eq,
    get_complexity_exactly_eq,
    get_triggers_exactly_eq,
)
from nucs.propagators.exactly_true_propagator import (
    compute_domains_exactly_true,
    get_complexity_exactly_true,
    get_triggers_exactly_true,
)
from nucs.propagators.gcc_propagator import compute_domains_gcc, get_complexity_gcc, get_triggers_gcc
from nucs.propagators.lexicographic_leq_propagator import (
    compute_domains_lexicographic_leq,
    get_complexity_lexicographic_leq,
    get_triggers_lexicographic_leq,
)
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_complexity_max_eq, get_triggers_max_eq
from nucs.propagators.max_leq_propagator import compute_domains_max_leq, get_complexity_max_leq, get_triggers_max_leq
from nucs.propagators.min_eq_propagator import compute_domains_min_eq, get_complexity_min_eq, get_triggers_min_eq
from nucs.propagators.min_geq_propagator import compute_domains_min_geq, get_complexity_min_geq, get_triggers_min_geq
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_triggers_relation,
)

GET_TRIGGERS_FCTS = []
GET_COMPLEXITY_FCTS = []
COMPUTE_DOMAINS_FCTS = []


def register_propagator(get_triggers_fct: Callable, get_complexity_fct: Callable, compute_domains_fct: Callable) -> int:
    """
    Register a propagator by adding its 3 functions to the corresponding lists of functions.
    :param get_triggers_fct: a function that returns the triggers
    :param get_complexity_fct: a function that computes the complexity
    :param compute_domains_fct: a function that computes the domains
    :return: the index of the propagator
    """
    GET_TRIGGERS_FCTS.append(get_triggers_fct)
    GET_COMPLEXITY_FCTS.append(get_complexity_fct)
    COMPUTE_DOMAINS_FCTS.append(compute_domains_fct)
    return len(COMPUTE_DOMAINS_FCTS) - 1


ALG_AND = register_propagator(get_triggers_and, get_complexity_and, compute_domains_and)
ALG_AFFINE_EQ = register_propagator(get_triggers_affine_eq, get_complexity_affine_eq, compute_domains_affine_eq)
ALG_AFFINE_GEQ = register_propagator(get_triggers_affine_geq, get_complexity_affine_geq, compute_domains_affine_geq)
ALG_AFFINE_LEQ = register_propagator(get_triggers_affine_leq, get_complexity_affine_leq, compute_domains_affine_leq)
ALG_ALLDIFFERENT = register_propagator(
    get_triggers_alldifferent, get_complexity_alldifferent, compute_domains_alldifferent
)
ALG_COUNT_EQ = register_propagator(get_triggers_count_eq, get_complexity_count_eq, compute_domains_count_eq)
ALG_DUMMY = register_propagator(get_triggers_dummy, get_complexity_dummy, compute_domains_dummy)
ALG_ELEMENT_LIV = register_propagator(get_triggers_element_liv, get_complexity_element_liv, compute_domains_element_liv)
ALG_ELEMENT_LIC = register_propagator(get_triggers_element_lic, get_complexity_element_lic, compute_domains_element_lic)
ALG_EXACTLY_EQ = register_propagator(get_triggers_exactly_eq, get_complexity_exactly_eq, compute_domains_exactly_eq)
ALG_EXACTLY_TRUE = register_propagator(
    get_triggers_exactly_true, get_complexity_exactly_true, compute_domains_exactly_true
)
ALG_GCC = register_propagator(get_triggers_gcc, get_complexity_gcc, compute_domains_gcc)
ALG_LEXICOGRAPHIC_LEQ = register_propagator(
    get_triggers_lexicographic_leq, get_complexity_lexicographic_leq, compute_domains_lexicographic_leq
)
ALG_MAX_EQ = register_propagator(get_triggers_max_eq, get_complexity_max_eq, compute_domains_max_eq)
ALG_MAX_LEQ = register_propagator(get_triggers_max_leq, get_complexity_max_leq, compute_domains_max_leq)
ALG_MIN_EQ = register_propagator(get_triggers_min_eq, get_complexity_min_eq, compute_domains_min_eq)
ALG_MIN_GEQ = register_propagator(get_triggers_min_geq, get_complexity_min_geq, compute_domains_min_geq)
ALG_RELATION = register_propagator(get_triggers_relation, get_complexity_relation, compute_domains_relation)


@njit(cache=True)
def pop_propagator(triggered_propagators: NDArray, previous_prop_idx: int) -> int:
    """
    Pops a propagator to be filtered.
    :param triggered_propagators: the candidate propagators
    :param previous_prop_idx: the index of the previous propagator which has been selected
    :return: an index
    """
    for prop_idx in range(len(triggered_propagators)):
        if triggered_propagators[prop_idx] and prop_idx != previous_prop_idx:
            triggered_propagators[prop_idx] = False
            return prop_idx
    return -1


@njit(cache=True)
def add_propagators(
    triggered_propagators: NDArray,
    not_entailed_propagators_stack: NDArray,
    stacks_top: NDArray,
    shr_domains_propagators: NDArray,
    dom_idx: int,
    bound: int,
) -> None:
    cp_top_idx = stacks_top[0]
    for prop_idx in range(len(triggered_propagators)):
        if shr_domains_propagators[dom_idx, bound, prop_idx] and not_entailed_propagators_stack[cp_top_idx, prop_idx]:
            triggered_propagators[prop_idx] = True
    # np.logical_or(triggered_propagators, shr_domains_propagators[dom_idx, bound], triggered_propagators)
    # np.logical_and(triggered_propagators, not_entailed_propagators, triggered_propagators)
