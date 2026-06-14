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
from typing import Callable, List

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_add, STORAGE_OFFSET
from nucs.propagators.abs_eq_propagator import compute_domains_abs_eq, get_complexity_abs_eq, get_triggers_abs_eq
from nucs.propagators.add_c_eq_propagator import (
    compute_domains_add_c_eq,
    get_complexity_add_c_eq,
    get_triggers_add_c_eq,
)
from nucs.propagators.linear_eq_c_propagator import (
    compute_domains_linear_eq_c,
    get_complexity_linear_eq_c,
    get_triggers_linear_eq_c,
)
from nucs.propagators.linear_geq_c_propagator import (
    compute_domains_linear_geq_c,
    get_complexity_linear_geq_c,
    get_triggers_linear_geq_c,
)
from nucs.propagators.linear_leq_c_propagator import (
    compute_domains_linear_leq_c,
    get_complexity_linear_leq_c,
    get_triggers_linear_leq_c,
)
from nucs.propagators.linear_neq_c_propagator import (
    compute_domains_linear_neq_c,
    get_complexity_linear_neq_c,
    get_triggers_linear_neq_c,
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
from nucs.propagators.count_geq_c_propagator import (
    compute_domains_count_geq_c,
    get_complexity_count_geq_c,
    get_triggers_count_geq_c,
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
from nucs.propagators.eq_propagator import compute_domains_eq, get_complexity_eq, get_triggers_eq
from nucs.propagators.eq_c_reif_propagator import (
    compute_domains_eq_c_reif,
    get_complexity_eq_c_reif,
    get_triggers_eq_c_reif,
)
from nucs.propagators.eq_reif_propagator import (
    compute_domains_eq_reif,
    get_complexity_eq_reif,
    get_triggers_eq_reif,
)
from nucs.propagators.gcc_propagator import compute_domains_gcc, get_complexity_gcc, get_triggers_gcc
from nucs.propagators.increasing_propagator import (
    compute_domains_increasing,
    get_complexity_increasing,
    get_triggers_increasing,
)
from nucs.propagators.inverse_propagator import (
    compute_domains_inverse,
    get_complexity_inverse,
    get_triggers_inverse,
)
from nucs.propagators.leq_c_propagator import compute_domains_leq_c, get_complexity_leq_c, get_triggers_leq_c
from nucs.propagators.leq_c_reif_propagator import (
    compute_domains_leq_c_reif,
    get_complexity_leq_c_reif,
    get_triggers_leq_c_reif,
)
from nucs.propagators.lexleq_propagator import (
    compute_domains_lexleq,
    get_complexity_lexleq,
    get_triggers_lexleq,
)
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_complexity_max_eq, get_triggers_max_eq
from nucs.propagators.max_leq_propagator import compute_domains_max_leq, get_complexity_max_leq, get_triggers_max_leq
from nucs.propagators.member_propagator import compute_domains_member, get_complexity_member, get_triggers_member
from nucs.propagators.min_eq_propagator import compute_domains_min_eq, get_complexity_min_eq, get_triggers_min_eq
from nucs.propagators.min_geq_propagator import compute_domains_min_geq, get_complexity_min_geq, get_triggers_min_geq
from nucs.propagators.mul_c_eq_propagator import (
    compute_domains_mul_c_eq,
    get_complexity_mul_c_eq,
    get_triggers_mul_c_eq,
)
from nucs.propagators.mul_eq_propagator import compute_domains_mul_eq, get_complexity_mul_eq, get_triggers_mul_eq
from nucs.propagators.neq_propagator import compute_domains_neq, get_complexity_neq, get_triggers_neq
from nucs.propagators.neq_reif_propagator import (
    compute_domains_neq_reif,
    get_complexity_neq_reif,
    get_triggers_neq_reif,
)
from nucs.propagators.no_sub_cycle_propagator import (
    compute_domains_no_sub_cycle,
    get_complexity_no_sub_cycle,
    get_triggers_no_sub_cycle,
)
from nucs.propagators.nvalue_propagator import (
    compute_domains_nvalue,
    get_complexity_nvalue,
    get_triggers_nvalue,
)
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_triggers_relation,
)
from nucs.propagators.scc_propagator import get_complexity_scc, get_triggers_scc, compute_domains_scc
from nucs.propagators.strictly_increasing_propagator import (
    compute_domains_strictly_increasing,
    get_complexity_strictly_increasing,
    get_triggers_strictly_increasing,
)
from nucs.propagators.subcircuit_propagator import (
    compute_domains_subcircuit,
    get_complexity_subcircuit,
    get_triggers_subcircuit,
)
from nucs.propagators.sum_eq_c_propagator import (
    compute_domains_sum_eq_c,
    get_complexity_sum_eq_c,
    get_triggers_sum_eq_c,
)
from nucs.propagators.sum_eq_propagator import compute_domains_sum_eq, get_complexity_sum_eq, get_triggers_sum_eq
from nucs.propagators.sum_geq_c_propagator import (
    compute_domains_sum_geq_c,
    get_complexity_sum_geq_c,
    get_triggers_sum_geq_c,
)
from nucs.propagators.sum_leq_c_propagator import (
    compute_domains_sum_leq_c,
    get_complexity_sum_leq_c,
    get_triggers_sum_leq_c,
)

GET_TRIGGERS_FCTS: List[Callable] = []
GET_COMPLEXITY_FCTS: List[Callable] = []
COMPUTE_DOMAINS_FCTS: List[Callable] = []


def get_algorithm_nb() -> int:
    return len(COMPUTE_DOMAINS_FCTS)


def register_propagator(get_triggers_fct: Callable, get_complexity_fct: Callable, compute_domains_fct: Callable) -> int:
    """
    Registers a propagator by adding its 3 functions to the corresponding lists of functions.

    :param get_triggers_fct: a function that returns the triggers
    :type get_triggers_fct: Callable
    :param get_complexity_fct: a function that computes the complexity
    :type get_complexity_fct: Callable
    :param compute_domains_fct: a function that computes the domains
    :type compute_domains_fct: Callable

    :return: the index of the propagator
    :rtype: int
    """
    GET_TRIGGERS_FCTS.append(get_triggers_fct)
    GET_COMPLEXITY_FCTS.append(get_complexity_fct)
    COMPUTE_DOMAINS_FCTS.append(compute_domains_fct)
    return get_algorithm_nb() - 1


ALG_ABS_EQ = register_propagator(get_triggers_abs_eq, get_complexity_abs_eq, compute_domains_abs_eq)
ALG_ADD_C_EQ = register_propagator(get_triggers_add_c_eq, get_complexity_add_c_eq, compute_domains_add_c_eq)
ALG_AND_EQ = register_propagator(get_triggers_and_eq, get_complexity_and_eq, compute_domains_and_eq)
ALG_LINEAR_EQ_C = register_propagator(get_triggers_linear_eq_c, get_complexity_linear_eq_c, compute_domains_linear_eq_c)
ALG_LINEAR_GEQ_C = register_propagator(
    get_triggers_linear_geq_c, get_complexity_linear_geq_c, compute_domains_linear_geq_c
)
ALG_LINEAR_LEQ_C = register_propagator(
    get_triggers_linear_leq_c, get_complexity_linear_leq_c, compute_domains_linear_leq_c
)
ALG_LINEAR_NEQ_C = register_propagator(
    get_triggers_linear_neq_c, get_complexity_linear_neq_c, compute_domains_linear_neq_c
)
ALG_ALLDIFFERENT = register_propagator(
    get_triggers_alldifferent, get_complexity_alldifferent, compute_domains_alldifferent
)
ALG_COUNT_EQ = register_propagator(get_triggers_count_eq, get_complexity_count_eq, compute_domains_count_eq)
ALG_COUNT_EQ_C = register_propagator(get_triggers_count_eq_c, get_complexity_count_eq_c, compute_domains_count_eq_c)
ALG_COUNT_GEQ_C = register_propagator(get_triggers_count_geq_c, get_complexity_count_geq_c, compute_domains_count_geq_c)
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
ALG_EQ = register_propagator(get_triggers_eq, get_complexity_eq, compute_domains_eq)
ALG_EQ_C_REIF = register_propagator(get_triggers_eq_c_reif, get_complexity_eq_c_reif, compute_domains_eq_c_reif)
ALG_EQ_REIF = register_propagator(get_triggers_eq_reif, get_complexity_eq_reif, compute_domains_eq_reif)
ALG_GCC = register_propagator(get_triggers_gcc, get_complexity_gcc, compute_domains_gcc)
ALG_INCREASING = register_propagator(get_triggers_increasing, get_complexity_increasing, compute_domains_increasing)
ALG_INVERSE = register_propagator(get_triggers_inverse, get_complexity_inverse, compute_domains_inverse)
ALG_LEQ_C = register_propagator(get_triggers_leq_c, get_complexity_leq_c, compute_domains_leq_c)
ALG_LEQ_C_REIF = register_propagator(get_triggers_leq_c_reif, get_complexity_leq_c_reif, compute_domains_leq_c_reif)
ALG_LEXLEQ = register_propagator(get_triggers_lexleq, get_complexity_lexleq, compute_domains_lexleq)
ALG_MAX_EQ = register_propagator(get_triggers_max_eq, get_complexity_max_eq, compute_domains_max_eq)
ALG_MAX_LEQ = register_propagator(get_triggers_max_leq, get_complexity_max_leq, compute_domains_max_leq)
ALG_MEMBER = register_propagator(get_triggers_member, get_complexity_member, compute_domains_member)
ALG_MIN_EQ = register_propagator(get_triggers_min_eq, get_complexity_min_eq, compute_domains_min_eq)
ALG_MIN_GEQ = register_propagator(get_triggers_min_geq, get_complexity_min_geq, compute_domains_min_geq)
ALG_MUL_C_EQ = register_propagator(get_triggers_mul_c_eq, get_complexity_mul_c_eq, compute_domains_mul_c_eq)
ALG_MUL_EQ = register_propagator(get_triggers_mul_eq, get_complexity_mul_eq, compute_domains_mul_eq)
ALG_NEQ = register_propagator(get_triggers_neq, get_complexity_neq, compute_domains_neq)
ALG_NEQ_REIF = register_propagator(get_triggers_neq_reif, get_complexity_neq_reif, compute_domains_neq_reif)
ALG_NO_SUB_CYCLE = register_propagator(
    get_triggers_no_sub_cycle, get_complexity_no_sub_cycle, compute_domains_no_sub_cycle
)
ALG_NVALUE = register_propagator(get_triggers_nvalue, get_complexity_nvalue, compute_domains_nvalue)
ALG_RELATION = register_propagator(get_triggers_relation, get_complexity_relation, compute_domains_relation)
ALG_SCC = register_propagator(get_triggers_scc, get_complexity_scc, compute_domains_scc)
ALG_STRICTLY_INCREASING = register_propagator(
    get_triggers_strictly_increasing, get_complexity_strictly_increasing, compute_domains_strictly_increasing
)
ALG_SUBCIRCUIT = register_propagator(get_triggers_subcircuit, get_complexity_subcircuit, compute_domains_subcircuit)
ALG_SUM_EQ = register_propagator(get_triggers_sum_eq, get_complexity_sum_eq, compute_domains_sum_eq)
ALG_SUM_EQ_C = register_propagator(get_triggers_sum_eq_c, get_complexity_sum_eq_c, compute_domains_sum_eq_c)
ALG_SUM_GEQ_C = register_propagator(get_triggers_sum_geq_c, get_complexity_sum_geq_c, compute_domains_sum_geq_c)
ALG_SUM_LEQ_C = register_propagator(get_triggers_sum_leq_c, get_complexity_sum_leq_c, compute_domains_sum_leq_c)


@njit(cache=True, fastmath=True)
def update_propagators(
    triggered_propagators: NDArray,
    entailed_propagator_depths: NDArray,
    triggers: NDArray,
    priorities: NDArray,
    propagator_nb: int,
) -> None:
    membership_offset = STORAGE_OFFSET + propagator_nb
    for prop_idx in triggers[1 : triggers[0] + 1]:
        if entailed_propagator_depths[prop_idx] == -1:
            buckets_add(triggered_propagators, priorities, prop_idx, membership_offset)
