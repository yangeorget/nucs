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
from typing import Any, Callable, List, Sequence

import numpy as np
from numba import njit, types  # type: ignore
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic
from numba.typed import List as NumbaList  # type: ignore
from numpy.typing import NDArray

from nucs.constants import NUMBA_DISABLE_JIT

# These per-search / per-propagator collections are Numba typed lists under the JIT and plain Python lists
# under NUMBA_DISABLE_JIT; both are indexed, iterated and measured the same way, so they are typed
# structurally as read-only sequences. The Callable element signatures mirror the matching SIGN_* in
# nucs.constants (under the JIT the element is actually a Numba FunctionType, which has no Python type).
NDArrayList = Sequence[NDArray]
ComputeDomainsFunctions = Sequence[Callable[[NDArray, NDArray], int]]
VariableHeuristicFunctions = Sequence[Callable[[NDArray, NDArray, int, NDArray], int]]
DomainHeuristicFunctions = Sequence[Callable[[NDArray, NDArray, NDArray, NDArray, NDArray, int, NDArray], int]]
ConsistencyAlgorithmFunctions = Sequence[Callable[..., int]]


@intrinsic
def function_ptr_from_address(typingctx, func_type_ref: types.FunctionType, addr: int):  # type: ignore
    """
    Recovers a function from FunctionType and address.
    """
    func_type = func_type_ref.instance_type  # type: ignore[attr-defined]

    def codegen(context, builder, sig, args):  # type: ignore
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)
        sfunc.c_addr = builder.inttoptr(args[1], context.get_value_type(types.voidptr))
        return sfunc._getvalue()

    return func_type(func_type_ref, addr), codegen


def addresses_from_functions(functions: List[Callable], signature: Any) -> NDArray:
    return (
        np.array([0])
        if NUMBA_DISABLE_JIT
        else np.array([_get_wrapper_address(function, signature) for function in functions])
    )


def build_typed_list(arrays: List[NDArray]) -> Any:
    """
    Builds a per-search list of arrays (the search's decision variables, or its heuristic parameters). Under
    the JIT a Numba typed list is returned (so the jitted search loop can index it by search), otherwise a
    plain Python list. The arrays may have different shapes; only their dtype and rank must match.
    """
    if NUMBA_DISABLE_JIT:
        return list(arrays)
    typed = NumbaList()
    for array in arrays:
        typed.append(array)
    return typed


def build_function_ptrs(functions: List[Callable], signature: Any) -> Any:
    """
    Materializes a typed list of function pointers of the given Numba FunctionType.

    Resolving a function object to its compiled-wrapper address is an interpreter-side operation, so it is
    done here in plain Python; the jitted :func:`_build_fcts` then rebuilds the first-class function pointers
    from those addresses. Built once at solver init so the inner loops avoid rebuilding it on every call.
    """
    return build_function_ptrs_from_addresses(
        addresses_from_functions(functions, signature), types.FunctionType(signature)
    )


@njit(cache=True)
def build_function_ptrs_from_addresses(addrs: NDArray, function_type: Any) -> NumbaList:
    """
    Rebuilds, in nopython mode, the typed list of first-class function pointers from their addresses.
    """
    function_ptrs = NumbaList.empty_list(function_type)
    for addr in addrs:
        function_ptrs.append(function_ptr_from_address(function_type, addr))  # type: ignore[call-arg, arg-type]
    return function_ptrs
