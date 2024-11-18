from typing import Optional

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import STATS_IDX_SOLVER_BACKTRACK_NB
from nucs.propagators.propagators import add_propagators


@njit(cache=True)
def cp_init(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    shr_domains_arr: NDArray,
) -> None:
    """
    Inits the choice points.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param shr_domains_arr: the shared domains
    :return: a Numpy array
    """
    shr_domains_stack[0] = shr_domains_arr
    not_entailed_propagators_stack[0] = True
    dom_update_stack[0, :] = 0
    stacks_top[0] = 0


@njit(cache=True)
def cp_put(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_index: int,
    bound: Optional[int] = 0,
) -> None:
    """
    Adds a choice point to the stack of choice points.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_index: the index of the shared domain which is changed
    :param bound: the bound (MIN or MAX) which is changed
    """
    dom_update_stack[stacks_top[0], 0] = dom_index
    dom_update_stack[stacks_top[0], 1] = bound
    shr_domains_stack[stacks_top[0] + 1, :, :] = shr_domains_stack[stacks_top[0], :, :]
    not_entailed_propagators_stack[stacks_top[0] + 1, :] = not_entailed_propagators_stack[stacks_top[0], :]
    stacks_top[0] += 1


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    triggered_propagators: NDArray,
    shr_domains_propagators: NDArray,
) -> bool:
    """
    Backtracks and updates the problem's domains.
    :param statistics: the statistics array
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :param shr_domains_propagators: a Numpy array indicating which propagators to trigger for a domain change
    :return: true iff it is possible to backtrack
    """
    if stacks_top[0] == 0:
        return False
    stacks_top[0] -= 1
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    add_propagators(
        triggered_propagators,
        not_entailed_propagators_stack,
        stacks_top,
        shr_domains_propagators,
        dom_update_stack[stacks_top[0], 0],
        1 - dom_update_stack[stacks_top[0], 1],
    )
    return True
