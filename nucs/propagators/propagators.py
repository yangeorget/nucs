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
from collections.abc import Callable, Sequence

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import STORAGE_OFFSET, buckets_add
from nucs.constants import EVENT_NB
from nucs.propagators.abs_eq_propagator import compute_domains_abs_eq, get_complexity_abs_eq, get_triggers_abs_eq
from nucs.propagators.add_c_eq_propagator import (
    compute_domains_add_c_eq,
    get_complexity_add_c_eq,
    get_triggers_add_c_eq,
)
from nucs.propagators.alldifferent_propagator import (
    compute_domains_alldifferent,
    get_complexity_alldifferent,
    get_triggers_alldifferent,
)
from nucs.propagators.and_eq_propagator import compute_domains_and_eq, get_complexity_and_eq, get_triggers_and_eq
from nucs.propagators.bin_packing_load_propagator import (
    compute_domains_bin_packing_load,
    get_complexity_bin_packing_load,
    get_triggers_bin_packing_load,
)
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
from nucs.propagators.cumulative_propagator import (
    compute_domains_cumulative,
    compute_domains_cumulative_var,
    get_complexity_cumulative,
    get_complexity_cumulative_var,
    get_triggers_cumulative,
    get_triggers_cumulative_var,
    is_vacuous_cumulative,
    is_vacuous_cumulative_var,
)
from nucs.propagators.diffn_propagator import (
    compute_domains_diffn,
    get_complexity_diffn,
    get_triggers_diffn,
)
from nucs.propagators.disjunctive_propagator import (
    compute_domains_disjunctive,
    get_complexity_disjunctive,
    get_triggers_disjunctive,
)
from nucs.propagators.div_c_eq_propagator import (
    compute_domains_div_c_eq,
    get_complexity_div_c_eq,
    get_triggers_div_c_eq,
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
from nucs.propagators.eq_c_imp_propagator import (
    compute_domains_eq_c_imp,
    get_complexity_eq_c_imp,
    get_triggers_eq_c_imp,
)
from nucs.propagators.eq_c_reif_propagator import (
    compute_domains_eq_c_reif,
    get_complexity_eq_c_reif,
    get_triggers_eq_c_reif,
)
from nucs.propagators.eq_imp_propagator import (
    compute_domains_eq_imp,
    get_complexity_eq_imp,
    get_triggers_eq_imp,
)
from nucs.propagators.eq_propagator import compute_domains_eq, get_complexity_eq, get_triggers_eq
from nucs.propagators.eq_reif_propagator import (
    compute_domains_eq_reif,
    get_complexity_eq_reif,
    get_triggers_eq_reif,
)
from nucs.propagators.gcc_propagator import (
    compute_domains_gcc,
    get_complexity_gcc,
    get_triggers_gcc,
    is_vacuous_gcc,
)
from nucs.propagators.if_then_else_propagator import (
    compute_domains_if_then_else,
    get_complexity_if_then_else,
    get_triggers_if_then_else,
)
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
from nucs.propagators.leq_c_imp_propagator import (
    compute_domains_leq_c_imp,
    get_complexity_leq_c_imp,
    get_triggers_leq_c_imp,
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
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_complexity_max_eq, get_triggers_max_eq
from nucs.propagators.member_propagator import compute_domains_member, get_complexity_member, get_triggers_member
from nucs.propagators.member_reif_propagator import (
    compute_domains_member_reif,
    get_complexity_member_reif,
    get_triggers_member_reif,
)
from nucs.propagators.min_eq_propagator import compute_domains_min_eq, get_complexity_min_eq, get_triggers_min_eq
from nucs.propagators.mod_c_eq_propagator import (
    compute_domains_mod_c_eq,
    get_complexity_mod_c_eq,
    get_triggers_mod_c_eq,
)
from nucs.propagators.mod_eq_propagator import compute_domains_mod_eq, get_complexity_mod_eq, get_triggers_mod_eq
from nucs.propagators.mul_c_eq_propagator import (
    compute_domains_mul_c_eq,
    get_complexity_mul_c_eq,
    get_triggers_mul_c_eq,
)
from nucs.propagators.mul_eq_propagator import compute_domains_mul_eq, get_complexity_mul_eq, get_triggers_mul_eq
from nucs.propagators.neq_c_reif_propagator import (
    compute_domains_neq_c_reif,
    get_complexity_neq_c_reif,
    get_triggers_neq_c_reif,
)
from nucs.propagators.neq_imp_propagator import (
    compute_domains_neq_imp,
    get_complexity_neq_imp,
    get_triggers_neq_imp,
)
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
from nucs.propagators.regular_propagator import (
    compute_domains_regular,
    get_complexity_regular,
    get_triggers_regular,
    is_vacuous_regular,
)
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_triggers_relation,
)
from nucs.propagators.scc_propagator import compute_domains_scc, get_complexity_scc, get_triggers_scc
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
from nucs.propagators.value_precede_propagator import (
    compute_domains_value_precede,
    get_complexity_value_precede,
    get_triggers_value_precede,
)

GET_TRIGGERS_FCTS: list[Callable] = []
GET_COMPLEXITY_FCTS: list[Callable] = []
COMPUTE_DOMAINS_FCTS: list[Callable] = []
IS_VACUOUS_FCTS: list[Callable] = []
# Whether one call of the algorithm reaches its own fixpoint, indexed by algorithm. A propagator that does
# not is rescheduled by the engine after any call that changed a domain, instead of iterating internally.
# Unlike the function lists above this is already the array the consistency algorithm reads, so it is passed
# to the jitted code as is rather than being rebuilt per problem or per solver.
IDEMPOTENT: NDArray = np.empty(0, dtype=np.bool_)


def is_never_vacuous(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the constraint is vacuous: the default answer, for the propagators this can never settle.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]
    :param domains: the initial domains, unused here
    :type domains: Sequence[tuple[int, int]]

    :return: False
    :rtype: bool
    """
    return False


def get_algorithm_nb() -> int:
    return len(COMPUTE_DOMAINS_FCTS)


def register_propagator(
    get_triggers_fct: Callable,
    get_complexity_fct: Callable,
    compute_domains_fct: Callable,
    is_vacuous_fct: Callable = is_never_vacuous,
    idempotent: bool = True,
) -> int:
    """
    Registers a propagator by adding its functions to the corresponding lists of functions.

    :param get_triggers_fct: a function that returns the triggers
    :type get_triggers_fct: Callable
    :param get_complexity_fct: a function that computes the complexity
    :type get_complexity_fct: Callable
    :param compute_domains_fct: a function that computes the domains
    :type compute_domains_fct: Callable
    :param is_vacuous_fct: a function that tells from the parameters and the initial domains whether the
        constraint is vacuous, in which case the propagator is not posted at all
    :type is_vacuous_fct: Callable
    :param idempotent: whether one call reaches the propagator's own fixpoint; when False the engine
        reschedules it after any call that changed a domain
    :type idempotent: bool

    :return: the index of the propagator
    :rtype: int
    """
    GET_TRIGGERS_FCTS.append(get_triggers_fct)
    GET_COMPLEXITY_FCTS.append(get_complexity_fct)
    COMPUTE_DOMAINS_FCTS.append(compute_domains_fct)
    IS_VACUOUS_FCTS.append(is_vacuous_fct)
    global IDEMPOTENT
    IDEMPOTENT = np.append(IDEMPOTENT, idempotent)
    return get_algorithm_nb() - 1


ALG_ABS_EQ = register_propagator(get_triggers_abs_eq, get_complexity_abs_eq, compute_domains_abs_eq)
ALG_ADD_C_EQ = register_propagator(get_triggers_add_c_eq, get_complexity_add_c_eq, compute_domains_add_c_eq)
ALG_AND_EQ = register_propagator(get_triggers_and_eq, get_complexity_and_eq, compute_domains_and_eq)
ALG_BIN_PACKING_LOAD = register_propagator(
    get_triggers_bin_packing_load,
    get_complexity_bin_packing_load,
    compute_domains_bin_packing_load,
    idempotent=False,
)
ALG_LINEAR_EQ_C = register_propagator(
    get_triggers_linear_eq_c, get_complexity_linear_eq_c, compute_domains_linear_eq_c, idempotent=False
)
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
ALG_CUMULATIVE = register_propagator(
    get_triggers_cumulative,
    get_complexity_cumulative,
    compute_domains_cumulative,
    is_vacuous_cumulative,
    idempotent=False,
)
ALG_CUMULATIVE_VAR = register_propagator(
    get_triggers_cumulative_var,
    get_complexity_cumulative_var,
    compute_domains_cumulative_var,
    is_vacuous_cumulative_var,
    idempotent=False,
)
ALG_DIFFN = register_propagator(get_triggers_diffn, get_complexity_diffn, compute_domains_diffn, idempotent=False)
ALG_DISJUNCTIVE = register_propagator(
    get_triggers_disjunctive, get_complexity_disjunctive, compute_domains_disjunctive, idempotent=False
)
ALG_DIV_C_EQ = register_propagator(get_triggers_div_c_eq, get_complexity_div_c_eq, compute_domains_div_c_eq)
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
ALG_EQ_C_IMP = register_propagator(get_triggers_eq_c_imp, get_complexity_eq_c_imp, compute_domains_eq_c_imp)
ALG_EQ_C_REIF = register_propagator(get_triggers_eq_c_reif, get_complexity_eq_c_reif, compute_domains_eq_c_reif)
ALG_EQ_IMP = register_propagator(get_triggers_eq_imp, get_complexity_eq_imp, compute_domains_eq_imp)
ALG_EQ_REIF = register_propagator(get_triggers_eq_reif, get_complexity_eq_reif, compute_domains_eq_reif)
ALG_GCC = register_propagator(get_triggers_gcc, get_complexity_gcc, compute_domains_gcc, is_vacuous_gcc)
ALG_IF_THEN_ELSE = register_propagator(
    get_triggers_if_then_else,
    get_complexity_if_then_else,
    compute_domains_if_then_else,
    idempotent=False,
)
ALG_INCREASING = register_propagator(get_triggers_increasing, get_complexity_increasing, compute_domains_increasing)
ALG_INVERSE = register_propagator(
    get_triggers_inverse, get_complexity_inverse, compute_domains_inverse, idempotent=False
)
ALG_LEQ_C = register_propagator(get_triggers_leq_c, get_complexity_leq_c, compute_domains_leq_c)
ALG_LEQ_C_IMP = register_propagator(get_triggers_leq_c_imp, get_complexity_leq_c_imp, compute_domains_leq_c_imp)
ALG_LEQ_C_REIF = register_propagator(get_triggers_leq_c_reif, get_complexity_leq_c_reif, compute_domains_leq_c_reif)
ALG_LEXLEQ = register_propagator(get_triggers_lexleq, get_complexity_lexleq, compute_domains_lexleq)
ALG_MAX_EQ = register_propagator(get_triggers_max_eq, get_complexity_max_eq, compute_domains_max_eq)
ALG_MEMBER = register_propagator(get_triggers_member, get_complexity_member, compute_domains_member)
ALG_MEMBER_REIF = register_propagator(get_triggers_member_reif, get_complexity_member_reif, compute_domains_member_reif)
ALG_MIN_EQ = register_propagator(get_triggers_min_eq, get_complexity_min_eq, compute_domains_min_eq)
ALG_MOD_C_EQ = register_propagator(get_triggers_mod_c_eq, get_complexity_mod_c_eq, compute_domains_mod_c_eq)
ALG_MOD_EQ = register_propagator(get_triggers_mod_eq, get_complexity_mod_eq, compute_domains_mod_eq, idempotent=False)
ALG_MUL_C_EQ = register_propagator(get_triggers_mul_c_eq, get_complexity_mul_c_eq, compute_domains_mul_c_eq)
ALG_MUL_EQ = register_propagator(get_triggers_mul_eq, get_complexity_mul_eq, compute_domains_mul_eq, idempotent=False)
ALG_NEQ = register_propagator(get_triggers_neq, get_complexity_neq, compute_domains_neq)
ALG_NEQ_IMP = register_propagator(get_triggers_neq_imp, get_complexity_neq_imp, compute_domains_neq_imp)
ALG_NEQ_C_REIF = register_propagator(get_triggers_neq_c_reif, get_complexity_neq_c_reif, compute_domains_neq_c_reif)
ALG_NEQ_REIF = register_propagator(get_triggers_neq_reif, get_complexity_neq_reif, compute_domains_neq_reif)
ALG_NO_SUB_CYCLE = register_propagator(
    get_triggers_no_sub_cycle, get_complexity_no_sub_cycle, compute_domains_no_sub_cycle, idempotent=False
)
ALG_NVALUE = register_propagator(get_triggers_nvalue, get_complexity_nvalue, compute_domains_nvalue)
ALG_REGULAR = register_propagator(
    get_triggers_regular,
    get_complexity_regular,
    compute_domains_regular,
    is_vacuous_regular,
    idempotent=False,
)
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
ALG_VALUE_PRECEDE = register_propagator(
    get_triggers_value_precede, get_complexity_value_precede, compute_domains_value_precede
)


@njit(cache=True)
def update_propagators(
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    variable: int,
    events: int,
) -> None:
    offset = (variable << EVENT_NB) | events
    membership_offset = STORAGE_OFFSET + len(priorities)
    for prop_idx in triggers[triggers_offsets[offset] : triggers_offsets[offset + 1]]:
        if not entailed[prop_idx]:
            buckets_add(triggered_propagators, priorities, prop_idx, membership_offset)
