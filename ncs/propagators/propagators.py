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
def init_propagators_to_filter(
    prop_to_filter: NDArray,
    shr_dom_changes: Optional[NDArray],
    prop_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
) -> None:
    if shr_dom_changes is None:  # this is an initialization
        prop_to_filter.fill(True)
    else:
        prop_to_filter.fill(False)
        for pidx in range(prop_nb):
            for var_idx in range(prop_var_bounds[pidx, START], prop_var_bounds[pidx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_dom_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_dom_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    prop_to_filter[pidx] = True
                    break


@jit(nopython=True, cache=True)
def update_propagators_to_filter(
    prop_to_filter: NDArray,
    shr_dom_changes: NDArray,
    prop_nb: int,
    prop_var_bounds: NDArray,
    prop_dom_indices: NDArray,
    prop_triggers: NDArray,
    prop_idx: int,
) -> None:
    for pidx in range(prop_nb):
        if pidx != prop_idx:
            for var_idx in range(prop_var_bounds[pidx, START], prop_var_bounds[pidx, END]):
                dom_idx = prop_dom_indices[var_idx]
                if (shr_dom_changes[dom_idx, MIN] and prop_triggers[var_idx, MIN]) or (
                    shr_dom_changes[dom_idx, MAX] and prop_triggers[var_idx, MAX]
                ):
                    prop_to_filter[pidx] = True
                    break
