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
"""
The builtin dispatch registry: each FlatZinc builtin name maps to a handler that emits one or more NuCS
propagators onto the model's problem.

Adding coverage for a new builtin is a single entry here (and, if it relies on a brand-new propagator, the
``/add-propagator`` flow plus, for kept globals, one body-less predicate file in the globals library).
"""

from typing import TYPE_CHECKING, Callable, Dict, List

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.parser import Id, Term
from nucs.propagators.propagators import (
    ALG_ABS_EQ,
    ALG_ADD_C_EQ,
    ALG_LINEAR_EQ_C,
    ALG_LINEAR_LEQ_C,
    ALG_ALLDIFFERENT,
    ALG_AND_EQ,
    ALG_EQ,
    ALG_EQ_REIF,
    ALG_EQ_C_REIF,
    ALG_ELEMENT_EQ,
    ALG_ELEMENT_L_EQ,
    ALG_GCC,
    ALG_LEQ_C,
    ALG_LEXLEQ,
    ALG_MAX_EQ,
    ALG_MIN_EQ,
    ALG_MUL_C_EQ,
    ALG_RELATION,
    ALG_SUM_EQ,
)

if TYPE_CHECKING:
    from nucs.fzn.model import FznModel

Handler = Callable[["FznModel", List[Term]], None]


def _is_const(model: "FznModel", term: Term) -> bool:
    """
    Returns whether a term resolves to a scalar integer constant.

    :param model: the model
    :type model: FznModel
    :param term: the term
    :type term: Term

    :return: True if the term is a constant
    :rtype: bool
    """
    if isinstance(term, (bool, int)):
        return True
    return isinstance(term, Id) and term.name in model.consts and isinstance(model.consts[term.name], int)


def _int_lin_eq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_eq(a, x, c)`` as the linear equality sum(a_i * x_i) = c.
    """
    coeffs = model.int_list_of(args[0])
    variables = model.var_list_of(args[1])
    model.problem.add_propagator(ALG_LINEAR_EQ_C, variables, coeffs + [model.const_of(args[2])])


def _int_lin_le(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_le(a, x, c)`` as the linear inequality sum(a_i * x_i) <= c.
    """
    coeffs = model.int_list_of(args[0])
    variables = model.var_list_of(args[1])
    model.problem.add_propagator(ALG_LINEAR_LEQ_C, variables, coeffs + [model.const_of(args[2])])


def _int_eq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_eq(x, y)`` as x - y = 0.
    """
    model.problem.add_propagator(
        ALG_LINEAR_EQ_C, [model.var_index_of(args[0]), model.var_index_of(args[1])], [1, -1, 0]
    )


def _int_eq_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_eq_reif(a, b, r)`` as the reification r <=> (a = b), mapping onto b <=> x = c when b is a
    constant and onto b <=> x = y otherwise.
    """
    reif = model.var_index_of(args[2])
    x = model.var_index_of(args[0])
    if _is_const(model, args[1]):
        model.problem.add_propagator(ALG_EQ_C_REIF, [reif, x], [model.const_of(args[1])])
    else:
        model.problem.add_propagator(ALG_EQ_REIF, [reif, x, model.var_index_of(args[1])])


def _bool2int(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool2int(b, x)`` as x = b; booleans are modelled as 0..1 integer variables.
    """
    model.problem.add_propagator(ALG_EQ, [model.var_index_of(args[1]), model.var_index_of(args[0])])


def _array_bool_and(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_bool_and(as, r)`` as the reification r <=> (and of the booleans in as).
    """
    variables = model.var_list_of(args[0]) + [model.var_index_of(args[1])]
    model.problem.add_propagator(ALG_AND_EQ, variables)


def _bool_and(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_and(a, b, r)`` as the reification r <=> (a and b).
    """
    model.problem.add_propagator(
        ALG_AND_EQ, [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
    )


def _bool_not(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_not(a, b)`` as a + b = 1, i.e. b is the negation of a; booleans are 0..1 integers.
    """
    model.problem.add_propagator(ALG_LINEAR_EQ_C, [model.var_index_of(args[0]), model.var_index_of(args[1])], [1, 1, 1])


def _bool_eq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_eq(a, b)`` as a = b.
    """
    model.problem.add_propagator(ALG_EQ, [model.var_index_of(args[0]), model.var_index_of(args[1])])


def _bool_le(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_le(a, b)`` as a <= b.
    """
    model.problem.add_propagator(ALG_LEQ_C, [model.var_index_of(args[0]), model.var_index_of(args[1])], [0])


def _int_le(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_le(x, y)`` as x <= y.
    """
    model.problem.add_propagator(ALG_LEQ_C, [model.var_index_of(args[0]), model.var_index_of(args[1])], [0])


def _int_lt(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lt(x, y)`` as x <= y - 1.
    """
    model.problem.add_propagator(ALG_LEQ_C, [model.var_index_of(args[0]), model.var_index_of(args[1])], [-1])


def _int_plus(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_plus(x, y, z)`` as x + y = z.
    """
    variables = [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
    model.problem.add_propagator(ALG_SUM_EQ, variables)


def _int_abs(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_abs(a, b)`` as abs(a) = b.
    """
    model.problem.add_propagator(ALG_ABS_EQ, [model.var_index_of(args[0]), model.var_index_of(args[1])])


def _int_times(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_times(x, y, z)`` as x * y = z, supported only when x or y is a constant.
    """
    if _is_const(model, args[1]):
        x, c, z = model.var_index_of(args[0]), model.const_of(args[1]), model.var_index_of(args[2])
    elif _is_const(model, args[0]):
        x, c, z = model.var_index_of(args[1]), model.const_of(args[0]), model.var_index_of(args[2])
    else:
        raise FznUnsupportedError("int_times with two variables is not supported (no var*var propagator)")
    model.problem.add_propagator(ALG_MUL_C_EQ, [x, z], [c])


def _int_max(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_max(x, y, z)`` as max(x, y) = z.
    """
    variables = [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
    model.problem.add_propagator(ALG_MAX_EQ, variables)


def _int_min(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_min(x, y, z)`` as min(x, y) = z.
    """
    variables = [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
    model.problem.add_propagator(ALG_MIN_EQ, variables)


def _all_different(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``all_different_int(x)``.
    """
    model.problem.add_propagator(ALG_ALLDIFFERENT, model.var_list_of(args[0]))


def _lex_lesseq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``lex_lesseq_int(x, y)`` as x <=_lex y.
    """
    variables = model.var_list_of(args[0]) + model.var_list_of(args[1])
    model.problem.add_propagator(ALG_LEXLEQ, variables)


def _table_int(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``table_int(x, t)``: each tuple of x must be a row of the table t (an extensional constraint).
    The 2D table is flattened row-major in FlatZinc, so it lines up with the RELATION tuple layout.
    """
    variables = model.var_list_of(args[0])
    model.problem.add_propagator(ALG_RELATION, variables, model.int_list_of(args[1]))


def _array_int_element(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_int_element(i, A, c)`` as A[i] = c, where A is an array of constants and i is 1-based.
    """
    index = model.var_index_of(args[0])
    array = model.int_list_of(args[1])
    value = model.var_index_of(args[2])
    index0 = model.problem.add_variable((0, len(array) - 1))
    model.problem.add_propagator(ALG_ADD_C_EQ, [index, index0], [-1])
    model.problem.add_propagator(ALG_ELEMENT_EQ, [index0, value], array)


def _array_var_int_element(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_var_int_element(i, X, c)`` as X[i] = c, where X is an array of variables and i is 1-based.
    """
    index = model.var_index_of(args[0])
    array = model.var_list_of(args[1])
    value = model.var_index_of(args[2])
    index0 = model.problem.add_variable((0, len(array) - 1))
    model.problem.add_propagator(ALG_ADD_C_EQ, [index, index0], [-1])
    model.problem.add_propagator(ALG_ELEMENT_L_EQ, array + [index0, value])


def _global_cardinality_low_up(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``global_cardinality_low_up(x, cover, lb, ub)``, supported only for a contiguous cover.
    """
    variables = model.var_list_of(args[0])
    cover = model.int_list_of(args[1])
    lb = model.int_list_of(args[2])
    ub = model.int_list_of(args[3])
    if cover != list(range(cover[0], cover[0] + len(cover))):
        raise FznUnsupportedError("global_cardinality with a non-contiguous cover is not supported")
    model.problem.add_propagator(ALG_GCC, variables, [cover[0]] + lb + ub)


BUILTINS: Dict[str, Handler] = {
    "all_different_int": _all_different,
    "array_bool_and": _array_bool_and,
    "array_int_element": _array_int_element,
    "array_var_int_element": _array_var_int_element,
    "bool2int": _bool2int,
    "bool_and": _bool_and,
    "bool_eq": _bool_eq,
    "bool_le": _bool_le,
    "bool_not": _bool_not,
    "fzn_all_different_int": _all_different,
    "fzn_global_cardinality_low_up": _global_cardinality_low_up,
    "fzn_lex_lesseq_int": _lex_lesseq,
    "global_cardinality_low_up": _global_cardinality_low_up,
    "int_abs": _int_abs,
    "int_eq": _int_eq,
    "int_eq_reif": _int_eq_reif,
    "int_le": _int_le,
    "int_lin_eq": _int_lin_eq,
    "int_lin_le": _int_lin_le,
    "int_lt": _int_lt,
    "int_max": _int_max,
    "int_min": _int_min,
    "int_plus": _int_plus,
    "int_times": _int_times,
    "lex_lesseq_int": _lex_lesseq,
    "nucs_table_int": _table_int,
}
