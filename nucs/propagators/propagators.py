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

from numba import int32, int64, njit, types, uint64  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import STORAGE_OFFSET, buckets_add
from nucs.constants import EVENT_NB, PROP_FLAG_IDEMPOTENT, PROP_FLAG_REPORTS_CHANGES
from nucs.numba_helper import NUMBA_DISABLE_JIT, function_ptr_from_address
from nucs.propagators.abs_eq_propagator import (
    compute_domains_abs_eq,
    get_complexity_abs_eq,
    get_triggers_abs_eq,
)
from nucs.propagators.add_c_eq_propagator import (
    compute_domains_add_c_eq,
    get_complexity_add_c_eq,
    get_triggers_add_c_eq,
)
from nucs.propagators.alldifferent_propagator import (
    compute_domains_alldifferent,
    get_complexity_alldifferent,
    get_state_alldifferent,
    get_triggers_alldifferent,
)
from nucs.propagators.and_eq_propagator import compute_domains_and_eq, get_complexity_and_eq, get_triggers_and_eq
from nucs.propagators.bin_packing_load_propagator import (
    compute_domains_bin_packing_load,
    get_complexity_bin_packing_load,
    get_state_bin_packing_load,
    get_triggers_bin_packing_load,
)
from nucs.propagators.circuit_chains_propagator import (
    compute_domains_circuit_chains,
    get_complexity_circuit_chains,
    get_state_circuit_chains,
    get_triggers_circuit_chains,
)
from nucs.propagators.count_eq_c_propagator import (
    compute_domains_count_eq_c,
    get_complexity_count_eq_c,
    get_state_count_eq_c,
    get_triggers_count_eq_c,
)
from nucs.propagators.count_eq_propagator import (
    compute_domains_count_eq,
    get_complexity_count_eq,
    get_state_count_eq,
    get_triggers_count_eq,
)
from nucs.propagators.count_geq_c_propagator import (
    compute_domains_count_geq_c,
    get_complexity_count_geq_c,
    get_state_count_geq_c,
    get_triggers_count_geq_c,
)
from nucs.propagators.count_leq_c_propagator import (
    compute_domains_count_leq_c,
    get_complexity_count_leq_c,
    get_state_count_leq_c,
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
    get_state_disjunctive,
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
    get_state_element_l_eq_alldifferent,
    get_triggers_element_l_eq_alldifferent,
)
from nucs.propagators.element_l_eq_c_alldifferent_propagator import (
    compute_domains_element_l_eq_c_alldifferent,
    get_complexity_element_l_eq_c_alldifferent,
    get_state_element_l_eq_c_alldifferent,
    get_triggers_element_l_eq_c_alldifferent,
)
from nucs.propagators.element_l_eq_c_propagator import (
    compute_domains_element_l_eq_c,
    get_complexity_element_l_eq_c,
    get_state_element_l_eq_c,
    get_triggers_element_l_eq_c,
)
from nucs.propagators.element_l_eq_propagator import (
    compute_domains_element_l_eq,
    get_complexity_element_l_eq,
    get_state_element_l_eq,
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
    get_state_gcc,
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
    get_state_inverse,
    get_triggers_inverse,
)
from nucs.propagators.leq_c_imp_propagator import (
    compute_domains_leq_c_imp,
    get_complexity_leq_c_imp,
    get_triggers_leq_c_imp,
)
from nucs.propagators.leq_c_propagator import (
    compute_domains_leq_c,
    get_complexity_leq_c,
    get_triggers_leq_c,
)
from nucs.propagators.leq_c_reif_propagator import (
    compute_domains_leq_c_reif,
    get_complexity_leq_c_reif,
    get_triggers_leq_c_reif,
)
from nucs.propagators.lexleq_propagator import (
    compute_domains_lexleq,
    get_complexity_lexleq,
    get_state_lexleq,
    get_triggers_lexleq,
)
from nucs.propagators.linear_eq_c_propagator import (
    compute_domains_linear_eq_c,
    get_complexity_linear_eq_c,
    get_state_linear_eq_c,
    get_triggers_linear_eq_c,
)
from nucs.propagators.linear_geq_c_propagator import (
    compute_domains_linear_geq_c,
    get_complexity_linear_geq_c,
    get_state_linear_geq_c,
    get_triggers_linear_geq_c,
)
from nucs.propagators.linear_leq_c_propagator import (
    compute_domains_linear_leq_c,
    get_complexity_linear_leq_c,
    get_state_linear_leq_c,
    get_triggers_linear_leq_c,
)
from nucs.propagators.linear_neq_c_propagator import (
    compute_domains_linear_neq_c,
    get_complexity_linear_neq_c,
    get_state_linear_neq_c,
    get_triggers_linear_neq_c,
)
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_complexity_max_eq, get_triggers_max_eq
from nucs.propagators.member_imp_propagator import (
    compute_domains_member_imp,
    get_complexity_member_imp,
    get_triggers_member_imp,
)
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
from nucs.propagators.neq_c_imp_propagator import (
    compute_domains_neq_c_imp,
    get_complexity_neq_c_imp,
    get_triggers_neq_c_imp,
)
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
from nucs.propagators.nvalue_propagator import (
    compute_domains_nvalue,
    get_complexity_nvalue,
    get_state_nvalue,
    get_triggers_nvalue,
)
from nucs.propagators.regular_propagator import (
    compute_domains_regular,
    get_complexity_regular,
    get_state_regular,
    get_triggers_regular,
    is_vacuous_regular,
)
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_state_relation,
    get_triggers_relation,
)
from nucs.propagators.strictly_increasing_propagator import (
    compute_domains_strictly_increasing,
    get_complexity_strictly_increasing,
    get_triggers_strictly_increasing,
)
from nucs.propagators.subcircuit_propagator import (
    compute_domains_subcircuit,
    get_complexity_subcircuit,
    get_state_subcircuit,
    get_triggers_subcircuit,
)
from nucs.propagators.sum_eq_c_propagator import (
    compute_domains_sum_eq_c,
    get_complexity_sum_eq_c,
    get_state_sum_eq_c,
    get_triggers_sum_eq_c,
)
from nucs.propagators.sum_eq_propagator import (
    compute_domains_sum_eq,
    get_complexity_sum_eq,
    get_state_sum_eq,
    get_triggers_sum_eq,
)
from nucs.propagators.sum_geq_c_propagator import (
    compute_domains_sum_geq_c,
    get_complexity_sum_geq_c,
    get_state_sum_geq_c,
    get_triggers_sum_geq_c,
)
from nucs.propagators.sum_leq_c_propagator import (
    compute_domains_sum_leq_c,
    get_complexity_sum_leq_c,
    get_state_sum_leq_c,
    get_triggers_sum_leq_c,
)
from nucs.propagators.value_precede_chain_propagator import (
    compute_domains_value_precede_chain,
    get_complexity_value_precede_chain,
    get_state_value_precede_chain,
    get_triggers_value_precede_chain,
)
from nucs.propagators.value_precede_propagator import (
    compute_domains_value_precede,
    get_complexity_value_precede,
    get_triggers_value_precede,
)

# The array arguments are typed C-contiguous (::1) rather than any-layout (:) so the hot loops in every
# propagator and in the consistency algorithm index with a plain offset instead of a stride multiply.
# All these arrays are contiguous np.empty/np.zeros/np.ones allocations threaded through unchanged.
# prop_state is this propagator's own slice of solver-owned memory (see get_state_default below): a
# trailed prefix of backtrackable cells followed by an untrailed hint suffix, both int32, both C-contiguous
# for the same reason domains and parameters are. Almost every propagator ignores it; alldifferent and gcc
# use the hint suffix as scratch space and, for alldifferent, as warm-started sort permutations.
SIGN_COMPUTE_DOMAINS = int64(int32[:, ::1], int32[::1], int32[::1])  # domains, parameters, prop_state
TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGN_COMPUTE_DOMAINS)

SIGN_GET_TRIGGERS = int64(uint64, uint64, int32[::1])
TYPE_GET_TRIGGERS = types.FunctionType(SIGN_GET_TRIGGERS)

GET_TRIGGERS_FCTS: list[Callable] = []
GET_COMPLEXITY_FCTS: list[Callable] = []
COMPUTE_DOMAINS_FCTS: list[Callable] = []
IS_VACUOUS_FCTS: list[Callable] = []
GET_STATE_FCTS: list[Callable] = []
# The PROP_FLAG_* properties of each algorithm, packed into one word and indexed by algorithm. Packed
# rather than one array per property because this is what a consistency algorithm receives and forwards:
# a second array would be a second parameter through SIGN_CONSISTENCY_ALG, and so a breaking change to
# every custom consistency algorithm, for something a spare bit already carries.
# A list, appended to like the five above, rather than the array the consistency algorithm wants:
# np.append returns a new array, so growing one would rebind this name, and any module that had imported
# it by value would keep an array one entry short of every algorithm registered since -- indexing past it
# for the new one. Problem.init makes the array, beside the algorithms that index it.
ALGORITHM_FLAGS: list[int] = []

# How a consistency algorithm calls a propagator: through the compiled address of its compute_domains, read
# out of an int64 array resolved once at solver init, rather than through a typed list of functions. Indexing
# a typed list needs Numba's reference-counting runtime, and bc_algorithm is compiled without it.
# Chosen at import because an address means nothing without the JIT: the pure-Python variant ignores it and
# calls the registered function. Inlined rather than cached on its own, so it adds no per-process cache load.
if NUMBA_DISABLE_JIT:

    def call_compute_domains(
        compute_domains_addrs: NDArray, algorithm: int, domains: NDArray, parameters: NDArray, prop_state: NDArray
    ) -> int:
        """
        Calls the compute_domains function of an algorithm, looked up in the registry.

        :param compute_domains_addrs: the compiled compute_domains address of each algorithm, unused without the JIT
        :type compute_domains_addrs: NDArray
        :param algorithm: the algorithm
        :type algorithm: int
        :param domains: the domains of the propagator's variables
        :type domains: NDArray
        :param parameters: the parameters of the propagator
        :type parameters: NDArray
        :param prop_state: the state block of the propagator
        :type prop_state: NDArray

        :return: the status returned by the propagator
        :rtype: int
        """
        return COMPUTE_DOMAINS_FCTS[algorithm](domains, parameters, prop_state)

else:

    @njit(cache=True, inline="always")
    def call_compute_domains(
        compute_domains_addrs: NDArray, algorithm: int, domains: NDArray, parameters: NDArray, prop_state: NDArray
    ) -> int:
        """
        Calls the compute_domains function of an algorithm through its compiled address.

        :param compute_domains_addrs: the compiled compute_domains address of each algorithm
        :type compute_domains_addrs: NDArray
        :param algorithm: the algorithm
        :type algorithm: int
        :param domains: the domains of the propagator's variables
        :type domains: NDArray
        :param parameters: the parameters of the propagator
        :type parameters: NDArray
        :param prop_state: the state block of the propagator
        :type prop_state: NDArray

        :return: the status returned by the propagator
        :rtype: int
        """
        return function_ptr_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[algorithm])(  # type: ignore[call-arg, arg-type]
            domains, parameters, prop_state
        )


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


def get_state_default(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the default for every propagator that needs none.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 0)
    :rtype: tuple[int, int]
    """
    return 0, 0


def get_algorithm_nb() -> int:
    return len(COMPUTE_DOMAINS_FCTS)


def get_algorithm_names() -> list[str]:
    """
    Returns the display name of each registered algorithm, indexed by algorithm.

    :return: the algorithm names
    :rtype: List[str]
    """
    return [fct.__name__.replace("compute_domains_", "").upper() for fct in COMPUTE_DOMAINS_FCTS]


def register_propagator(
    get_triggers_fct: Callable,
    get_complexity_fct: Callable,
    compute_domains_fct: Callable,
    is_vacuous_fct: Callable = is_never_vacuous,
    idempotent: bool = True,
    get_state_fct: Callable = get_state_default,
    reports_changes: bool = False,
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
    :param get_state_fct: a function that returns the (trailed_nb, hint_nb) size of this propagator's
        state block, defaulting to none
    :type get_state_fct: Callable
    :param reports_changes: whether the propagator writes, into the first cell of its state block's hint
        suffix, whether it changed any domain -- which lets the engine skip the write-back scan on the
        calls that changed none. Opting in obliges get_state_fct to reserve that cell
    :type reports_changes: bool

    :return: the index of the propagator
    :rtype: int
    """
    GET_TRIGGERS_FCTS.append(get_triggers_fct)
    GET_COMPLEXITY_FCTS.append(get_complexity_fct)
    COMPUTE_DOMAINS_FCTS.append(compute_domains_fct)
    IS_VACUOUS_FCTS.append(is_vacuous_fct)
    GET_STATE_FCTS.append(get_state_fct)
    ALGORITHM_FLAGS.append(
        (PROP_FLAG_IDEMPOTENT if idempotent else 0) | (PROP_FLAG_REPORTS_CHANGES if reports_changes else 0)
    )
    return get_algorithm_nb() - 1


ALG_ABS_EQ = register_propagator(get_triggers_abs_eq, get_complexity_abs_eq, compute_domains_abs_eq)
ALG_ADD_C_EQ = register_propagator(get_triggers_add_c_eq, get_complexity_add_c_eq, compute_domains_add_c_eq)
ALG_AND_EQ = register_propagator(get_triggers_and_eq, get_complexity_and_eq, compute_domains_and_eq)
ALG_BIN_PACKING_LOAD = register_propagator(
    get_triggers_bin_packing_load,
    get_complexity_bin_packing_load,
    compute_domains_bin_packing_load,
    idempotent=False,
    get_state_fct=get_state_bin_packing_load,
)
ALG_LINEAR_EQ_C = register_propagator(
    get_triggers_linear_eq_c,
    get_complexity_linear_eq_c,
    compute_domains_linear_eq_c,
    get_state_fct=get_state_linear_eq_c,
    reports_changes=True,
)
ALG_LINEAR_GEQ_C = register_propagator(
    get_triggers_linear_geq_c,
    get_complexity_linear_geq_c,
    compute_domains_linear_geq_c,
    get_state_fct=get_state_linear_geq_c,
    reports_changes=True,
)
ALG_LINEAR_LEQ_C = register_propagator(
    get_triggers_linear_leq_c,
    get_complexity_linear_leq_c,
    compute_domains_linear_leq_c,
    get_state_fct=get_state_linear_leq_c,
    reports_changes=True,
)
ALG_LINEAR_NEQ_C = register_propagator(
    get_triggers_linear_neq_c,
    get_complexity_linear_neq_c,
    compute_domains_linear_neq_c,
    get_state_fct=get_state_linear_neq_c,
    reports_changes=True,
)
ALG_ALLDIFFERENT = register_propagator(
    get_triggers_alldifferent,
    get_complexity_alldifferent,
    compute_domains_alldifferent,
    get_state_fct=get_state_alldifferent,
    reports_changes=True,
)
ALG_CIRCUIT_CHAINS = register_propagator(
    get_triggers_circuit_chains,
    get_complexity_circuit_chains,
    compute_domains_circuit_chains,
    get_state_fct=get_state_circuit_chains,
)
ALG_COUNT_EQ = register_propagator(
    get_triggers_count_eq,
    get_complexity_count_eq,
    compute_domains_count_eq,
    get_state_fct=get_state_count_eq,
    reports_changes=True,
)
ALG_COUNT_EQ_C = register_propagator(
    get_triggers_count_eq_c,
    get_complexity_count_eq_c,
    compute_domains_count_eq_c,
    get_state_fct=get_state_count_eq_c,
)
ALG_COUNT_GEQ_C = register_propagator(
    get_triggers_count_geq_c,
    get_complexity_count_geq_c,
    compute_domains_count_geq_c,
    get_state_fct=get_state_count_geq_c,
    reports_changes=True,
)
ALG_COUNT_LEQ_C = register_propagator(
    get_triggers_count_leq_c,
    get_complexity_count_leq_c,
    compute_domains_count_leq_c,
    get_state_fct=get_state_count_leq_c,
    reports_changes=True,
)
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
    get_triggers_disjunctive,
    get_complexity_disjunctive,
    compute_domains_disjunctive,
    idempotent=False,
    get_state_fct=get_state_disjunctive,
)
ALG_DIV_C_EQ = register_propagator(get_triggers_div_c_eq, get_complexity_div_c_eq, compute_domains_div_c_eq)
ALG_DUMMY = register_propagator(get_triggers_dummy, get_complexity_dummy, compute_domains_dummy)
ALG_ELEMENT_EQ = register_propagator(get_triggers_element_eq, get_complexity_element_eq, compute_domains_element_eq)
ALG_ELEMENT_L_EQ = register_propagator(
    get_triggers_element_l_eq,
    get_complexity_element_l_eq,
    compute_domains_element_l_eq,
    get_state_fct=get_state_element_l_eq,
    reports_changes=True,
)
ALG_ELEMENT_L_EQ_ALLDIFFERENT = register_propagator(
    get_triggers_element_l_eq_alldifferent,
    get_complexity_element_l_eq_alldifferent,
    compute_domains_element_l_eq_alldifferent,
    get_state_fct=get_state_element_l_eq_alldifferent,
    reports_changes=True,
)
ALG_ELEMENT_L_EQ_C = register_propagator(
    get_triggers_element_l_eq_c,
    get_complexity_element_l_eq_c,
    compute_domains_element_l_eq_c,
    get_state_fct=get_state_element_l_eq_c,
    reports_changes=True,
)
ALG_ELEMENT_L_EQ_C_ALLDIFFERENT = register_propagator(
    get_triggers_element_l_eq_c_alldifferent,
    get_complexity_element_l_eq_c_alldifferent,
    compute_domains_element_l_eq_c_alldifferent,
    get_state_fct=get_state_element_l_eq_c_alldifferent,
    reports_changes=True,
)
ALG_EQ = register_propagator(get_triggers_eq, get_complexity_eq, compute_domains_eq)
ALG_EQ_C_IMP = register_propagator(get_triggers_eq_c_imp, get_complexity_eq_c_imp, compute_domains_eq_c_imp)
ALG_EQ_C_REIF = register_propagator(get_triggers_eq_c_reif, get_complexity_eq_c_reif, compute_domains_eq_c_reif)
ALG_EQ_IMP = register_propagator(get_triggers_eq_imp, get_complexity_eq_imp, compute_domains_eq_imp)
ALG_EQ_REIF = register_propagator(get_triggers_eq_reif, get_complexity_eq_reif, compute_domains_eq_reif)
ALG_GCC = register_propagator(
    get_triggers_gcc,
    get_complexity_gcc,
    compute_domains_gcc,
    is_vacuous_gcc,
    get_state_fct=get_state_gcc,
    reports_changes=True,
)
ALG_IF_THEN_ELSE = register_propagator(
    get_triggers_if_then_else, get_complexity_if_then_else, compute_domains_if_then_else, idempotent=False
)
ALG_INCREASING = register_propagator(get_triggers_increasing, get_complexity_increasing, compute_domains_increasing)
ALG_INVERSE = register_propagator(
    get_triggers_inverse,
    get_complexity_inverse,
    compute_domains_inverse,
    idempotent=False,
    get_state_fct=get_state_inverse,
    reports_changes=True,
)
ALG_LEQ_C = register_propagator(get_triggers_leq_c, get_complexity_leq_c, compute_domains_leq_c)
ALG_LEQ_C_IMP = register_propagator(get_triggers_leq_c_imp, get_complexity_leq_c_imp, compute_domains_leq_c_imp)
ALG_LEQ_C_REIF = register_propagator(get_triggers_leq_c_reif, get_complexity_leq_c_reif, compute_domains_leq_c_reif)
ALG_LEXLEQ = register_propagator(
    get_triggers_lexleq,
    get_complexity_lexleq,
    compute_domains_lexleq,
    get_state_fct=get_state_lexleq,
    reports_changes=True,
)
ALG_MAX_EQ = register_propagator(get_triggers_max_eq, get_complexity_max_eq, compute_domains_max_eq)
ALG_MEMBER = register_propagator(get_triggers_member, get_complexity_member, compute_domains_member)
ALG_MEMBER_IMP = register_propagator(get_triggers_member_imp, get_complexity_member_imp, compute_domains_member_imp)
ALG_MEMBER_REIF = register_propagator(get_triggers_member_reif, get_complexity_member_reif, compute_domains_member_reif)
ALG_MIN_EQ = register_propagator(get_triggers_min_eq, get_complexity_min_eq, compute_domains_min_eq)
ALG_MOD_C_EQ = register_propagator(get_triggers_mod_c_eq, get_complexity_mod_c_eq, compute_domains_mod_c_eq)
ALG_MOD_EQ = register_propagator(get_triggers_mod_eq, get_complexity_mod_eq, compute_domains_mod_eq, idempotent=False)
ALG_MUL_C_EQ = register_propagator(get_triggers_mul_c_eq, get_complexity_mul_c_eq, compute_domains_mul_c_eq)
ALG_MUL_EQ = register_propagator(get_triggers_mul_eq, get_complexity_mul_eq, compute_domains_mul_eq)
ALG_NEQ = register_propagator(get_triggers_neq, get_complexity_neq, compute_domains_neq)
ALG_NEQ_IMP = register_propagator(get_triggers_neq_imp, get_complexity_neq_imp, compute_domains_neq_imp)
ALG_NEQ_C_IMP = register_propagator(get_triggers_neq_c_imp, get_complexity_neq_c_imp, compute_domains_neq_c_imp)
ALG_NEQ_C_REIF = register_propagator(get_triggers_neq_c_reif, get_complexity_neq_c_reif, compute_domains_neq_c_reif)
ALG_NEQ_REIF = register_propagator(get_triggers_neq_reif, get_complexity_neq_reif, compute_domains_neq_reif)
ALG_NVALUE = register_propagator(
    get_triggers_nvalue,
    get_complexity_nvalue,
    compute_domains_nvalue,
    get_state_fct=get_state_nvalue,
    reports_changes=True,
)
ALG_REGULAR = register_propagator(
    get_triggers_regular,
    get_complexity_regular,
    compute_domains_regular,
    is_vacuous_regular,
    idempotent=False,
    get_state_fct=get_state_regular,
    reports_changes=True,
)
ALG_RELATION = register_propagator(
    get_triggers_relation,
    get_complexity_relation,
    compute_domains_relation,
    get_state_fct=get_state_relation,
    reports_changes=True,
)
ALG_STRICTLY_INCREASING = register_propagator(
    get_triggers_strictly_increasing, get_complexity_strictly_increasing, compute_domains_strictly_increasing
)
ALG_SUBCIRCUIT = register_propagator(
    get_triggers_subcircuit,
    get_complexity_subcircuit,
    compute_domains_subcircuit,
    get_state_fct=get_state_subcircuit,
)
ALG_SUM_EQ = register_propagator(
    get_triggers_sum_eq,
    get_complexity_sum_eq,
    compute_domains_sum_eq,
    get_state_fct=get_state_sum_eq,
    reports_changes=True,
)
ALG_SUM_EQ_C = register_propagator(
    get_triggers_sum_eq_c,
    get_complexity_sum_eq_c,
    compute_domains_sum_eq_c,
    get_state_fct=get_state_sum_eq_c,
    reports_changes=True,
)
ALG_SUM_GEQ_C = register_propagator(
    get_triggers_sum_geq_c,
    get_complexity_sum_geq_c,
    compute_domains_sum_geq_c,
    get_state_fct=get_state_sum_geq_c,
    reports_changes=True,
)
ALG_SUM_LEQ_C = register_propagator(
    get_triggers_sum_leq_c,
    get_complexity_sum_leq_c,
    compute_domains_sum_leq_c,
    get_state_fct=get_state_sum_leq_c,
    reports_changes=True,
)
ALG_VALUE_PRECEDE = register_propagator(
    get_triggers_value_precede, get_complexity_value_precede, compute_domains_value_precede
)
ALG_VALUE_PRECEDE_CHAIN = register_propagator(
    get_triggers_value_precede_chain,
    get_complexity_value_precede_chain,
    compute_domains_value_precede_chain,
    get_state_fct=get_state_value_precede_chain,
    reports_changes=True,
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
