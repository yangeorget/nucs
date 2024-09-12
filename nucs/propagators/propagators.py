import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.memory import MAX, MIN
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


@njit(cache=True)
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
def update_triggered_propagators(
    triggered_propagators: NDArray,
    not_entailed_propagators: NDArray,
    shr_domain_changes: NDArray,
    shr_domains_propagators: NDArray,
    previous_prop_idx: int,
) -> None:
    for shr_domain_idx in range(len(shr_domain_changes)):
        if shr_domain_changes[shr_domain_idx, MIN]:
            np.logical_or(triggered_propagators, shr_domains_propagators[shr_domain_idx, MIN], triggered_propagators)
        if shr_domain_changes[shr_domain_idx, MAX]:
            np.logical_or(triggered_propagators, shr_domains_propagators[shr_domain_idx, MAX], triggered_propagators)
    np.logical_and(triggered_propagators, not_entailed_propagators, triggered_propagators)
    if previous_prop_idx != -1:
        triggered_propagators[previous_prop_idx] = False


@njit(cache=True)
def pop_propagator(triggered_propagators: NDArray) -> int:
    for prop_idx, triggered_prop in enumerate(triggered_propagators):
        if triggered_prop:
            triggered_propagators[prop_idx] = False
            return prop_idx
    return -1
