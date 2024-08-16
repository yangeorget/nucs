from numba import jit  # type: ignore
from numpy._typing import NDArray

from ncs.memory import END, MAX, MIN, START
from ncs.propagators import (
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_propagator,
    count_eq_propagator,
    dummy_propagator,
    exactly_eq_propagator,
    max_propagator,
    min_propagator,
)

ALG_AFFINE_EQ = 0
ALG_AFFINE_GEQ = 1
ALG_AFFINE_LEQ = 2
ALG_ALLDIFFERENT = 3
ALG_COUNT_EQ = 4
ALG_DUMMY = 5
ALG_EXACTLY_EQ = 6
ALG_MAX = 7
ALG_MIN = 8


TRIGGER_MODULES = [
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_propagator,
    count_eq_propagator,
    dummy_propagator,
    exactly_eq_propagator,
    max_propagator,
    min_propagator,
]


def get_triggers(algorithm: int, size: int, data: NDArray) -> NDArray:
    return TRIGGER_MODULES[algorithm].get_triggers(size, data)


@jit(
    "boolean(uint8, int32[::1,:], int32[:])",
    nopython=True,
    cache=True,
)
def compute_domains(algorithm: int, domains: NDArray, data: NDArray) -> bool:
    """
    Computes the new domains for the variables.
    :param domains: the initial domains of the variables
    :return: the new domains or None if an inconsistency is detected
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
    if algorithm == ALG_MAX:
        return max_propagator.compute_domains(domains, data)
    return min_propagator.compute_domains(domains, data)


@jit(nopython=True, cache=True)
def init_propagator_queue(
    propagator_queue: NDArray,
    shr_domain_changes: NDArray,
    propagator_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
) -> None:
    propagator_queue.fill(False)
    for prop_idx in range(propagator_nb):
        for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
            dom_idx = prop_dom_indices[var_idx]
            if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
            ):
                propagator_queue[prop_idx] = True
                break


@jit(nopython=True, cache=True)
def update_propagator_queue(
    propagator_queue: NDArray,
    shr_domain_changes: NDArray,
    propagator_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
    previous_prop_idx: int,
) -> None:
    for prop_idx in range(propagator_nb):
        if prop_idx != previous_prop_idx and not propagator_queue[prop_idx]:
            for var_idx in range(prop_var_bounds[prop_idx, START], prop_var_bounds[prop_idx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_domain_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_domain_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    propagator_queue[prop_idx] = True
                    break
