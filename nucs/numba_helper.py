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
from typing import List

from numba import types  # type: ignore
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic


@intrinsic
def function_from_address(typingctx, func_type_ref: types.FunctionType, addr: int):  # type: ignore
    """
    Recovers a function from FunctionType and address.
    """
    func_type = func_type_ref.instance_type

    def codegen(context, builder, sig, args):  # type: ignore
        _, addr = args
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)
        sfunc.addr = builder.inttoptr(addr, context.get_value_type(types.voidptr))
        return sfunc._getvalue()

    return func_type(func_type_ref, addr), codegen


def build_function_address_list(functions, signature) -> List[int]:  # type: ignore
    return [_get_wrapper_address(function, signature) for function in functions]
