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
    stacks_height: NDArray,
    shr_domains_arr: NDArray,
) -> None:
    shr_domains_stack[0] = shr_domains_arr
    not_entailed_propagators_stack[0] = True
    dom_update_stack[0, :] = 0
    stacks_height[0] = 1


@njit(cache=True)
def cp_put(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_height: NDArray,
    dom_index: int,
    bound: Optional[int] = 0,
) -> None:
    shr_domains_stack[stacks_height[0], :, :] = shr_domains_stack[0, :, :]
    not_entailed_propagators_stack[stacks_height[0], :] = not_entailed_propagators_stack[0, :]
    dom_update_stack[stacks_height[0], 0] = dom_index
    dom_update_stack[stacks_height[0], 1] = bound
    stacks_height[0] += 1


@njit(cache=True)
def cp_pop(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_height: NDArray,
) -> bool:
    if stacks_height[0] == 1:
        return False
    stacks_height[0] -= 1
    shr_domains_stack[0, :, :] = shr_domains_stack[stacks_height[0], :, :]
    not_entailed_propagators_stack[0, :] = not_entailed_propagators_stack[stacks_height[0], :]
    dom_update_stack[0] = dom_update_stack[stacks_height[0]]
    return True


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_height: NDArray,
    triggered_propagators: NDArray,
    shr_domains_propagators: NDArray,
) -> bool:
    """
    Backtracks and updates the problem's domains.
    :param statistics: the statistics array
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param stacks_height: the height of both stacks
    :return: true iff it is possible to backtrack
    """
    if not cp_pop(shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_height):
        return False
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    add_propagators(
        triggered_propagators,
        not_entailed_propagators_stack[0],
        shr_domains_propagators,
        dom_update_stack[0, 0],
        1 - dom_update_stack[0, 1],
    )
    return True
