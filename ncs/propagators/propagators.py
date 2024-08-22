from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import END, MAX, MIN, START
from ncs.propagators import (
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_propagator,
    count_eq_propagator,
    dummy_propagator,
    exactly_eq_propagator,
    lexicographic_leq_propagator,
    max_eq_propagator,
    max_leq_propagator,
    min_eq_propagator,
    min_geq_propagator,
)

# The ordinals of the algorithms for all propagators (sorted by alphabetical ordering).
ALG_AFFINE_EQ = 0
ALG_AFFINE_GEQ = 1
ALG_AFFINE_LEQ = 2
ALG_ALLDIFFERENT = 3
ALG_COUNT_EQ = 4
ALG_DUMMY = 5
ALG_EXACTLY_EQ = 6
ALG_LEXICOGRAPHIC_LEQ = 7
ALG_MAX_EQ = 8
ALG_MAX_LEQ = 9
ALG_MIN_EQ = 10
ALG_MIN_GEQ = 11

PROPAGATOR_MODULES = [
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_propagator,
    count_eq_propagator,
    dummy_propagator,
    exactly_eq_propagator,
    lexicographic_leq_propagator,
    max_eq_propagator,
    max_leq_propagator,
    min_eq_propagator,
    min_geq_propagator,
]


def get_triggers(algorithm: int, size: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for a propagator.
    :param algorithm: the ordinal of the algorithm
    :param size: the number of variables
    :return: an array of triggers
    """
    return PROPAGATOR_MODULES[algorithm].get_triggers(size, data)


@jit("int64(uint8, int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(algorithm: int, domains: NDArray, data: NDArray) -> int:
    """
    Computes the new domains for the variables.
    :param algorithm: the ordinal of the algorithm
    :param domains: the initial domains of the variables
    :param data: the parameters of the propagator
    :return: a status as an integer (INCONSISTENCY, CONSISTENCY, ENTAILMENT)
    """
    if algorithm == ALG_AFFINE_EQ:
        return affine_eq_propagator.compute_domains(domains, data)
    if algorithm == ALG_AFFINE_GEQ:
        return affine_geq_propagator.compute_domains(domains, data)
    if algorithm == ALG_AFFINE_LEQ:
        return affine_leq_propagator.compute_domains(domains, data)
    if algorithm == ALG_ALLDIFFERENT:
        return alldifferent_propagator.compute_domains(domains, data)
    if algorithm == ALG_COUNT_EQ:
        return count_eq_propagator.compute_domains(domains, data)
    if algorithm == ALG_DUMMY:
        return dummy_propagator.compute_domains(domains, data)
    if algorithm == ALG_EXACTLY_EQ:
        return exactly_eq_propagator.compute_domains(domains, data)
    if algorithm == ALG_LEXICOGRAPHIC_LEQ:
        return lexicographic_leq_propagator.compute_domains(domains, data)
    if algorithm == ALG_MAX_EQ:
        return max_eq_propagator.compute_domains(domains, data)
    if algorithm == ALG_MAX_LEQ:
        return max_leq_propagator.compute_domains(domains, data)
    if algorithm == ALG_MIN_EQ:
        return min_eq_propagator.compute_domains(domains, data)
    return min_geq_propagator.compute_domains(domains, data)


@jit(nopython=True, cache=True)
def init_triggered_propagators(
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    shr_domain_changes: NDArray,
    propagator_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
) -> None:
    triggered_propagators.fill(False)
    for prop_idx in range(propagator_nb):
        if not entailed_propagators[prop_idx]:
            for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    triggered_propagators[prop_idx] = True
                    break


@jit(nopython=True, cache=True)
def update_triggered_propagators(
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    shr_domain_changes: NDArray,
    propagator_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
    previous_prop_idx: int,
) -> None:
    for prop_idx in range(propagator_nb):
        if not entailed_propagators[prop_idx] and prop_idx != previous_prop_idx and not triggered_propagators[prop_idx]:
            for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    triggered_propagators[prop_idx] = True
                    break
