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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import Callable

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.heaps import min_heap_add
from nucs.propagators.abs_eq_propagator import compute_domains_abs_eq, get_complexity_abs_eq, get_triggers_abs_eq
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
from nucs.propagators.and_eq_propagator import compute_domains_and_eq, get_complexity_and_eq, get_triggers_and_eq
from nucs.propagators.count_eq_c_propagator import (
    compute_domains_count_eq_c,
    get_complexity_count_eq_c,
    get_triggers_count_eq_c,
)
from nucs.propagators.count_eq_propagator import (
    compute_domains_count_eq,
    get_complexity_count_eq,
    get_triggers_count_eq,
)
from nucs.propagators.count_leq_c_propagator import (
    compute_domains_count_leq_c,
    get_complexity_count_leq_c,
    get_triggers_count_leq_c,
)
from nucs.propagators.dummy_propagator import compute_domains_dummy, get_complexity_dummy, get_triggers_dummy
from nucs.propagators.element_eq_propagator import (
    compute_domains_element_eq,
    get_complexity_element_eq,
    get_triggers_element_eq,
)
from nucs.propagators.element_l_eq_alldifferent_propagator import (
    compute_domains_element_l_eq_alldifferent,
    get_complexity_element_l_eq_alldifferent,
    get_triggers_element_l_eq_alldifferent,
)
from nucs.propagators.element_l_eq_c_alldifferent_propagator import (
    compute_domains_element_l_eq_c_alldifferent,
    get_complexity_element_l_eq_c_alldifferent,
    get_triggers_element_l_eq_c_alldifferent,
)
from nucs.propagators.element_l_eq_c_propagator import (
    compute_domains_element_l_eq_c,
    get_complexity_element_l_eq_c,
    get_triggers_element_l_eq_c,
)
from nucs.propagators.element_l_eq_propagator import (
    compute_domains_element_l_eq,
    get_complexity_element_l_eq,
    get_triggers_element_l_eq,
)
from nucs.propagators.equiv_eq_propagator import (
    compute_domains_equiv_eq,
    get_complexity_equiv_eq,
    get_triggers_equiv_eq,
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
from nucs.propagators.no_sub_cycle_propagator import (
    compute_domains_no_sub_cycle,
    get_complexity_no_sub_cycle,
    get_triggers_no_sub_cycle,
)
from nucs.propagators.permutation_aux_propagator import (
    compute_domains_permutation_aux,
    get_complexity_permutation_aux,
    get_triggers_permutation_aux,
)
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_triggers_relation,
)
from nucs.propagators.scc_propagator import compute_domains_scc, get_complexity_scc, get_triggers_scc

GET_TRIGGERS_FCTS = []
GET_COMPLEXITY_FCTS = []
COMPUTE_DOMAINS_FCTS = []


def register_propagator(get_triggers_fct: Callable, get_complexity_fct: Callable, compute_domains_fct: Callable) -> int:
    """
    Registers a propagator by adding its 3 functions to the corresponding lists of functions.
    :param get_triggers_fct: a function that returns the triggers
    :param get_complexity_fct: a function that computes the complexity
    :param compute_domains_fct: a function that computes the domains
    :return: the index of the propagator
    """
    GET_TRIGGERS_FCTS.append(get_triggers_fct)
    GET_COMPLEXITY_FCTS.append(get_complexity_fct)
    COMPUTE_DOMAINS_FCTS.append(compute_domains_fct)
    return len(COMPUTE_DOMAINS_FCTS) - 1


ALG_ABS_EQ = register_propagator(get_triggers_abs_eq, get_complexity_abs_eq, compute_domains_abs_eq)
ALG_AND_EQ = register_propagator(get_triggers_and_eq, get_complexity_and_eq, compute_domains_and_eq)
ALG_AFFINE_EQ = register_propagator(get_triggers_affine_eq, get_complexity_affine_eq, compute_domains_affine_eq)
ALG_AFFINE_GEQ = register_propagator(get_triggers_affine_geq, get_complexity_affine_geq, compute_domains_affine_geq)
ALG_AFFINE_LEQ = register_propagator(get_triggers_affine_leq, get_complexity_affine_leq, compute_domains_affine_leq)
ALG_ALLDIFFERENT = register_propagator(
    get_triggers_alldifferent, get_complexity_alldifferent, compute_domains_alldifferent
)
ALG_COUNT_EQ = register_propagator(get_triggers_count_eq, get_complexity_count_eq, compute_domains_count_eq)
ALG_COUNT_EQ_C = register_propagator(get_triggers_count_eq_c, get_complexity_count_eq_c, compute_domains_count_eq_c)
ALG_COUNT_LEQ_C = register_propagator(get_triggers_count_leq_c, get_complexity_count_leq_c, compute_domains_count_leq_c)
ALG_DUMMY = register_propagator(get_triggers_dummy, get_complexity_dummy, compute_domains_dummy)
ALG_ELEMENT_EQ = register_propagator(get_triggers_element_eq, get_complexity_element_eq, compute_domains_element_eq)
ALG_ELEMENT_L_EQ = register_propagator(
    get_triggers_element_l_eq, get_complexity_element_l_eq, compute_domains_element_l_eq
)
ALG_ELEMENT_L_EQ_ALLDIFFERENT = register_propagator(
    get_triggers_element_l_eq_alldifferent,
    get_complexity_element_l_eq_alldifferent,
    compute_domains_element_l_eq_alldifferent,
)
ALG_ELEMENT_L_EQ_C = register_propagator(
    get_triggers_element_l_eq_c, get_complexity_element_l_eq_c, compute_domains_element_l_eq_c
)
ALG_ELEMENT_L_EQ_C_ALLDIFFERENT = register_propagator(
    get_triggers_element_l_eq_c_alldifferent,
    get_complexity_element_l_eq_c_alldifferent,
    compute_domains_element_l_eq_c_alldifferent,
)
ALG_EQUIV_EQ = register_propagator(get_triggers_equiv_eq, get_complexity_equiv_eq, compute_domains_equiv_eq)
ALG_GCC = register_propagator(get_triggers_gcc, get_complexity_gcc, compute_domains_gcc)
ALG_LEXICOGRAPHIC_LEQ = register_propagator(
    get_triggers_lexicographic_leq, get_complexity_lexicographic_leq, compute_domains_lexicographic_leq
)
ALG_MAX_EQ = register_propagator(get_triggers_max_eq, get_complexity_max_eq, compute_domains_max_eq)
ALG_MAX_LEQ = register_propagator(get_triggers_max_leq, get_complexity_max_leq, compute_domains_max_leq)
ALG_MIN_EQ = register_propagator(get_triggers_min_eq, get_complexity_min_eq, compute_domains_min_eq)
ALG_MIN_GEQ = register_propagator(get_triggers_min_geq, get_complexity_min_geq, compute_domains_min_geq)
ALG_NO_SUB_CYCLE = register_propagator(
    get_triggers_no_sub_cycle, get_complexity_no_sub_cycle, compute_domains_no_sub_cycle
)
ALG_PERMUTATION_AUX = register_propagator(
    get_triggers_permutation_aux, get_complexity_permutation_aux, compute_domains_permutation_aux
)
ALG_RELATION = register_propagator(get_triggers_relation, get_complexity_relation, compute_domains_relation)
ALG_SCC = register_propagator(get_triggers_scc, get_complexity_scc, compute_domains_scc)


@njit(cache=True)
def reset_triggered_propagators(triggered_propagators: NDArray, propagator_nb: int) -> None:
    triggered_propagators[:] = 0
    for prop_idx in range(propagator_nb):
        min_heap_add(triggered_propagators, propagator_nb, prop_idx)


@njit(cache=True)
def update_propagators(
    triggered_propagators: NDArray,
    not_entailed_propagators: NDArray,
    triggers: NDArray,
    events: int,
    dom_idx: int,
    previous_prop_idx: int = -1,
) -> None:
    propagator_nb = len(not_entailed_propagators)
    for prop_idx in triggers[dom_idx, events]:
        if prop_idx == -1:
            break
        if not_entailed_propagators[prop_idx] and prop_idx != previous_prop_idx:
            min_heap_add(triggered_propagators, propagator_nb, prop_idx)
