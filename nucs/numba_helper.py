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
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numba import njit, types  # type: ignore
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic
from numba.typed import List as NumbaList  # type: ignore
from numpy.typing import NDArray

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")

# These function tables are Numba typed lists under the JIT and plain Python lists under
# NUMBA_DISABLE_JIT; both are indexed and measured the same way, so they are typed structurally as
# read-only sequences. The Callable element signatures mirror the matching SIGN_* in nucs.constants
# (under the JIT the element is actually a Numba FunctionType, which has no Python type).
ComputeDomainsFunctions = Sequence[Callable[[NDArray, NDArray], int]]
VariableHeuristicFunctions = Sequence[Callable[[NDArray, NDArray, NDArray], int]]
DomainHeuristicFunctions = Sequence[Callable[[NDArray, int, NDArray], tuple[int, int]]]
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


def addresses_from_functions(
    functions: list[Callable], signature: Any, used: NDArray | None = None, filler_index: int | None = None
) -> NDArray:
    """
    Returns the compiled-wrapper addresses of the given functions.

    Resolving an address forces Numba to materialize the compiled code, which is costly: when used is provided,
    only those indices are resolved and the remaining slots share the address of the filler function
    (which is never called but keeps the table indexable by any function index).

    :param functions: the functions
    :type functions: List[Callable]
    :param signature: the Numba signature of the functions
    :type signature: Any
    :param used: the indices of the functions to resolve, all of them if None
    :type used: Optional[NDArray]
    :param filler_index: the index of the function whose address fills the unresolved slots,
                         if None all the functions are resolved
    :type filler_index: Optional[int]

    :return: the addresses
    :rtype: NDArray
    """
    if NUMBA_DISABLE_JIT:
        return np.array([0])
    if used is None or filler_index is None:
        return np.array([_get_wrapper_address(function, signature) for function in functions])
    addresses = np.full(len(functions), _get_wrapper_address(functions[filler_index], signature), dtype=np.int64)
    for index in used:
        addresses[index] = _get_wrapper_address(functions[index], signature)
    return addresses


def build_function_ptrs(
    functions: list[Callable], signature: Any, used: NDArray | None = None, filler_index: int | None = None
) -> Any:
    """
    Materializes a typed list of function pointers of the given Numba FunctionType.

    Resolving a function object to its compiled-wrapper address is an interpreter-side operation, so it is
    done here in plain Python; the jitted :func:`_build_fcts` then rebuilds the first-class function pointers
    from those addresses. Built once at solver init so the inner loops avoid rebuilding it on every call.
    When used is provided, only those indices are resolved (see :func:`addresses_from_functions`).
    """
    return build_function_ptrs_from_addresses(
        addresses_from_functions(functions, signature, used, filler_index), types.FunctionType(signature)
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
