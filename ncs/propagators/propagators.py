from typing import Optional

from numba import jit  # type: ignore
from numpy._typing import NDArray

from ncs.propagators import (
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_lopez_ortiz_propagator,
    dummy_propagator,
)
from ncs.utils import END, MAX, MIN, START

ALG_AFFINE_EQ = 0
ALG_AFFINE_GEQ = 1
ALG_AFFINE_LEQ = 2
ALG_ALLDIFFERENT = 3
ALG_DUMMY = 4


def get_triggers(algorithm: int, size: int, data: NDArray) -> NDArray:
    if algorithm == ALG_AFFINE_EQ:
        return affine_eq_propagator.get_triggers(size, data)
    elif algorithm == ALG_AFFINE_GEQ:
        return affine_geq_propagator.get_triggers(size, data)
    elif algorithm == ALG_AFFINE_LEQ:
        return affine_leq_propagator.get_triggers(size, data)
    elif algorithm == ALG_ALLDIFFERENT:
        return alldifferent_lopez_ortiz_propagator.get_triggers(size, data)
    else:
        return dummy_propagator.get_triggers(size, data)


@jit(nopython=True, cache=True)
def compute_domains(algorithm: int, domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    Computes the new domains for the variables.
    :param domains: the initial domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    if algorithm == ALG_AFFINE_EQ:
        return affine_eq_propagator.compute_domains(domains, data)
    elif algorithm == ALG_AFFINE_GEQ:
        return affine_geq_propagator.compute_domains(domains, data)
    elif algorithm == ALG_AFFINE_LEQ:
        return affine_leq_propagator.compute_domains(domains, data)
    elif algorithm == ALG_ALLDIFFERENT:
        return alldifferent_lopez_ortiz_propagator.compute_domains(domains, data)
    else:
        return dummy_propagator.compute_domains(domains, data)


@jit(nopython=True, cache=True)
def init_propagator_queue(
    propagator_queue: NDArray,
    shr_domain_changes: Optional[NDArray],
    propagator_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
) -> None:
    if shr_domain_changes is None:  # this is an initialization
        propagator_queue.fill(True)
    else:
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
def update_propagators_to_filter(
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
