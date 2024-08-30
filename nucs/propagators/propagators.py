from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.memory import END, MAX, MIN, START
from nucs.propagators.affine_eq_propagator import compute_domains_affine_eq, get_triggers_affine_eq
from nucs.propagators.affine_geq_propagator import compute_domains_affine_geq, get_triggers_affine_geq
from nucs.propagators.affine_leq_propagator import compute_domains_affine_leq, get_triggers_affine_leq
from nucs.propagators.alldifferent_propagator import compute_domains_alldifferent, get_triggers_alldifferent
from nucs.propagators.count_eq_propagator import compute_domains_count_eq, get_triggers_count_eq
from nucs.propagators.dummy_propagator import compute_domains_dummy, get_triggers_dummy
from nucs.propagators.element_lic_propagator import compute_domains_element_lic, get_triggers_element_lic
from nucs.propagators.element_liv_propagator import compute_domains_element_liv, get_triggers_element_liv
from nucs.propagators.exactly_eq_propagator import compute_domains_exactly_eq, get_triggers_exactly_eq
from nucs.propagators.lexicographic_leq_propagator import (
    compute_domains_lexicographic_leq,
    get_triggers_lexicographic_leq,
)
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_triggers_max_eq
from nucs.propagators.max_leq_propagator import compute_domains_max_leq, get_triggers_max_leq
from nucs.propagators.min_eq_propagator import compute_domains_min_eq, get_triggers_min_eq
from nucs.propagators.min_geq_propagator import compute_domains_min_geq, get_triggers_min_geq
from nucs.propagators.relation_propagator import compute_domains_relation, get_triggers_relation

# The ordinals of the algorithms for all propagators (sorted by alphabetical ordering).
(
    ALG_AFFINE_EQ,
    ALG_AFFINE_GEQ,
    ALG_AFFINE_LEQ,
    ALG_ALLDIFFERENT,
    ALG_COUNT_EQ,
    ALG_DUMMY,
    ALG_ELEMENT_LIV,
    ALG_ELEMENT_LIC,
    ALG_EXACTLY_EQ,
    ALG_LEXICOGRAPHIC_LEQ,
    ALG_MAX_EQ,
    ALG_MAX_LEQ,
    ALG_MIN_EQ,
    ALG_MIN_GEQ,
    ALG_RELATION,
) = tuple(range(15))


GET_TRIGGERS_FUNCTIONS = [
    get_triggers_affine_eq,
    get_triggers_affine_geq,
    get_triggers_affine_leq,
    get_triggers_alldifferent,
    get_triggers_count_eq,
    get_triggers_dummy,
    get_triggers_element_liv,
    get_triggers_element_lic,
    get_triggers_exactly_eq,
    get_triggers_lexicographic_leq,
    get_triggers_max_eq,
    get_triggers_max_leq,
    get_triggers_min_eq,
    get_triggers_min_geq,
    get_triggers_relation,
]


@njit("int64(uint8, int32[::1,:], int32[:])", cache=True)
def compute_domains(algorithm: int, domains: NDArray, data: NDArray) -> int:
    """
    Computes the new domains for the variables.
    :param algorithm: the ordinal of the algorithm
    :param domains: the initial domains of the variables
    :param data: the parameters of the propagator
    :return: a status as an integer (INCONSISTENCY, CONSISTENCY, ENTAILMENT)
    """
    if algorithm == ALG_AFFINE_EQ:
        return compute_domains_affine_eq(domains, data)
    if algorithm == ALG_AFFINE_GEQ:
        return compute_domains_affine_geq(domains, data)
    if algorithm == ALG_AFFINE_LEQ:
        return compute_domains_affine_leq(domains, data)
    if algorithm == ALG_ALLDIFFERENT:
        return compute_domains_alldifferent(domains, data)
    if algorithm == ALG_COUNT_EQ:
        return compute_domains_count_eq(domains, data)
    if algorithm == ALG_DUMMY:
        return compute_domains_dummy(domains, data)
    if algorithm == ALG_ELEMENT_LIV:
        return compute_domains_element_liv(domains, data)
    if algorithm == ALG_ELEMENT_LIC:
        return compute_domains_element_lic(domains, data)
    if algorithm == ALG_EXACTLY_EQ:
        return compute_domains_exactly_eq(domains, data)
    if algorithm == ALG_LEXICOGRAPHIC_LEQ:
        return compute_domains_lexicographic_leq(domains, data)
    if algorithm == ALG_MAX_EQ:
        return compute_domains_max_eq(domains, data)
    if algorithm == ALG_MAX_LEQ:
        return compute_domains_max_leq(domains, data)
    if algorithm == ALG_MIN_EQ:
        return compute_domains_min_eq(domains, data)
    if algorithm == ALG_MIN_GEQ:
        return compute_domains_min_geq(domains, data)
    return compute_domains_relation(domains, data)


@njit(cache=True)
def init_triggered_propagators(
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    shr_domain_changes: NDArray,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
) -> None:
    triggered_propagators.fill(False)
    for prop_idx, entailed_propagator in enumerate(entailed_propagators):
        if not entailed_propagator:
            for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    triggered_propagators[prop_idx] = True
                    break


@njit(cache=True)
def update_triggered_propagators(
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    shr_domain_changes: NDArray,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
    previous_prop_idx: int,
) -> None:
    for prop_idx, entailed_propagator in enumerate(entailed_propagators):
        if not entailed_propagator and prop_idx != previous_prop_idx and not triggered_propagators[prop_idx]:
            for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    triggered_propagators[prop_idx] = True
                    break
