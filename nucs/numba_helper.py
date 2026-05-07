###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
from typing import List, Callable, Any

import numpy as np
from numba import njit, types  # type: ignore
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic
from numba.typed import List as NumbaList  # type: ignore
from numpy.typing import NDArray

from nucs.constants import NUMBA_DISABLE_JIT
from nucs.constants import (
    TYPE_COMPUTE_DOMAINS,
    TYPE_CONSISTENCY_ALG,
    TYPE_DOM_HEURISTIC,
    TYPE_VAR_HEURISTIC,
)


@intrinsic
def function_from_address(typingctx, func_type_ref: types.FunctionType, addr: int):  # type: ignore
    """
    Recovers a function from FunctionType and address.
    """
    func_type = func_type_ref.instance_type

    def codegen(context, builder, sig, args):  # type: ignore
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)
        sfunc.c_addr = builder.inttoptr(args[1], context.get_value_type(types.voidptr))
        return sfunc._getvalue()

    return func_type(func_type_ref, addr), codegen


def address_from_function(function: Callable, signature: Any) -> Any:
    return _get_wrapper_address(function, signature)


def addresses_from_functions(functions: List[Callable], signature: Any) -> NDArray:
    return (
        np.array([0])
        if NUMBA_DISABLE_JIT
        else np.array([address_from_function(function, signature) for function in functions])
    )


@njit(cache=True)
def build_compute_domains_fcts(compute_domains_addrs: NDArray) -> NumbaList:
    """
    Materializes the typed list of compute_domains function pointers from their addresses.
    Built once at solver init so the BC inner loop avoids rebuilding it on every call.
    """
    fcts = NumbaList.empty_list(TYPE_COMPUTE_DOMAINS)
    for alg_idx in range(len(compute_domains_addrs)):
        fcts.append(function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[alg_idx]))
    return fcts


@njit(cache=True)
def build_consistency_alg_fcts(addr: int) -> NumbaList:
    """
    Recovers a consistency-algorithm function from its address, wrapped in a 1-element typed list.
    The list is the parent object — returning the bare FunctionType across the Python boundary
    triggers a "parent object not set" MemoryError. Index [0] once inside the JIT function to use.
    """
    fcts = NumbaList.empty_list(TYPE_CONSISTENCY_ALG)
    fcts.append(function_from_address(TYPE_CONSISTENCY_ALG, addr))
    return fcts


@njit(cache=True)
def build_var_heuristic_fcts(addr: int) -> NumbaList:
    """
    Recovers a variable-heuristic function from its address.
    """
    fcts = NumbaList.empty_list(TYPE_VAR_HEURISTIC)
    fcts.append(function_from_address(TYPE_VAR_HEURISTIC, addr))
    return fcts


@njit(cache=True)
def build_dom_heuristic_fcts(addr: int) -> NumbaList:
    """
    Recovers a domain-heuristic function from its address.
    """
    fcts = NumbaList.empty_list(TYPE_DOM_HEURISTIC)
    fcts.append(function_from_address(TYPE_DOM_HEURISTIC, addr))
    return fcts
