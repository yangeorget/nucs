from typing import Optional

import numpy as np
from numba import jit
from numpy._typing import NDArray

from ncs.propagators import (
    affine_eq_propagator,
    affine_geq_propagator,
    affine_leq_propagator,
    alldifferent_lopez_ortiz_propagator,
    dummy_propagator,
)
from ncs.utils import END, START

ALG_AFFINE_EQ = 0
ALG_AFFINE_GEQ = 1
ALG_AFFINE_LEQ = 2
ALG_ALLDIFFERENT = 3
ALG_DUMMY = 4


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
    elif algorithm == ALG_DUMMY:
        return dummy_propagator.compute_domains(domains, data)
    return None


@jit(nopython=True, cache=True)
def init_propagators_to_filter(
    propagators_to_filter: NDArray,
    changes: Optional[NDArray],
    propagator_nb: int,
    propagator_bounds: NDArray,
    propagator_indices: NDArray,
) -> None:
    if changes is None:  # this is an initialization
        propagators_to_filter.fill(True)
    else:
        for pidx in range(propagator_nb):
            propagators_to_filter[pidx] = np.any(
                changes[propagator_indices[propagator_bounds[pidx, START] : propagator_bounds[pidx, END]]]
            )


@jit(nopython=True, cache=True)
def update_propagators_to_filter(
    propagators_to_filter: NDArray,
    changes: NDArray,
    propagator_nb: int,
    propagator_bounds: NDArray,
    propagator_indices: NDArray,
    propagator_idx: int,
) -> None:
    for pidx in range(propagator_nb):
        if pidx != propagator_idx:
            if np.any(changes[propagator_indices[propagator_bounds[pidx, START] : propagator_bounds[pidx, END]]]):
                propagators_to_filter[pidx] = True
