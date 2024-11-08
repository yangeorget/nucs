from numba import njit  # type: ignore
from numpy._typing import NDArray


@njit(cache=True)
def cp_init(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    stacks_height: NDArray,
    shr_domains_arr: NDArray,
) -> None:
    shr_domains_stack[0] = shr_domains_arr
    not_entailed_propagators_stack[0, :] = True
    stacks_height[0] = 1


@njit(cache=True)
def cp_put(shr_domains_stack: NDArray, not_entailed_propagators_stack: NDArray, stacks_height: NDArray) -> None:
    shr_domains_stack[stacks_height[0], :, :] = shr_domains_stack[0, :, :]
    not_entailed_propagators_stack[stacks_height[0], :] = not_entailed_propagators_stack[0, :]
    stacks_height[0] += 1


@njit(cache=True)
def cp_pop(shr_domains_stack: NDArray, not_entailed_propagators_stack: NDArray, stacks_height: NDArray) -> bool:
    if stacks_height[0] == 1:
        return False
    stacks_height[0] -= 1
    shr_domains_stack[0, :, :] = shr_domains_stack[stacks_height[0], :, :]
    not_entailed_propagators_stack[0, :] = not_entailed_propagators_stack[stacks_height[0], :]
    return True
