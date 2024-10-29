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
# Copyright 2024 - Yan Georget
###############################################################################
import os

from numba import types  # type: ignore
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic


@intrinsic
def function_from_address(typingctx, func_type_ref, addr):  # type: ignore
    """
    Recovers a function from FunctionType and address.
    """
    func_type = func_type_ref.instance_type

    def codegen(context, builder, sig, args):  # type: ignore
        _, addr = args
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)
        llty = context.get_value_type(types.voidptr)
        addr_ptr = builder.inttoptr(addr, llty)
        sfunc.addr = addr_ptr
        return sfunc._getvalue()

    sig = func_type(func_type_ref, addr)
    return sig, codegen


def build_function_address_list(fcts, signature):  # type: ignore
    return [_get_wrapper_address(fct, signature) for fct in fcts]


NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")
