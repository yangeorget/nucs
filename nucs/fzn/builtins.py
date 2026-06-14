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
    ALG_ALLDIFFERENT,
    ALG_AND_EQ,
    ALG_COUNT_EQ,
    ALG_COUNT_EQ_C,
    ALG_COUNT_GEQ_C,
    ALG_COUNT_LEQ_C,
    ALG_ELEMENT_EQ,
    ALG_ELEMENT_L_EQ,
    ALG_ELEMENT_L_EQ_C,
    ALG_EQ,
    ALG_EQ_C_REIF,
    ALG_EQ_REIF,
    ALG_GCC,
    ALG_INCREASING,
    ALG_INVERSE,
    ALG_LEQ_C,
    ALG_LEQ_C_REIF,
    ALG_LEXLEQ,
    ALG_LINEAR_EQ_C,
    ALG_LINEAR_GEQ_C,
    ALG_LINEAR_LEQ_C,
    ALG_LINEAR_NEQ_C,
    ALG_MAX_EQ,
    ALG_MEMBER,
    ALG_MIN_EQ,
    ALG_MUL_C_EQ,
    ALG_MUL_EQ,
    ALG_NEQ,
    ALG_NEQ_REIF,
    ALG_NO_SUB_CYCLE,
    ALG_NVALUE,
    ALG_RELATION,
    ALG_STRICTLY_INCREASING,
    ALG_SUBCIRCUIT,
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


def _int_lin_ne(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_ne(a, x, c)`` as the linear disequality sum(a_i * x_i) != c.
    """
    coeffs = model.int_list_of(args[0])
    variables = model.var_list_of(args[1])
    model.problem.add_propagator(ALG_LINEAR_NEQ_C, variables, coeffs + [model.const_of(args[2])])


def _aux_lin_sum(model: "FznModel", coeffs: List[int], variables: List[int]) -> int:
    """
    Creates an auxiliary variable s bound to sum(a_i * x_i) and returns its index. Used to reduce a
    reified linear constraint to a reification on a single variable.
    """
    lo = hi = 0
    for a, v in zip(coeffs, variables):
        v_min, v_max = model.problem.domains[v]
        if a >= 0:
            lo += a * v_min
            hi += a * v_max
        else:
            lo += a * v_max
            hi += a * v_min
    s = model.problem.add_variable((lo, hi))
    model.problem.add_propagator(ALG_LINEAR_EQ_C, variables + [s], coeffs + [-1, 0])
    return s


def _int_lin_eq_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_eq_reif(a, x, c, r)`` as r <=> sum(a_i * x_i) = c, via an auxiliary sum variable.
    """
    s = _aux_lin_sum(model, model.int_list_of(args[0]), model.var_list_of(args[1]))
    model.problem.add_propagator(ALG_EQ_C_REIF, [model.var_index_of(args[3]), s], [model.const_of(args[2])])


def _int_lin_le_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_le_reif(a, x, c, r)`` as r <=> sum(a_i * x_i) <= c, via an auxiliary sum variable.
    """
    s = _aux_lin_sum(model, model.int_list_of(args[0]), model.var_list_of(args[1]))
    model.problem.add_propagator(ALG_LEQ_C_REIF, [model.var_index_of(args[3]), s, model.var_index_of(args[2])], [0])


def _int_lin_ne_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_ne_reif(a, x, c, r)`` as r <=> sum(a_i * x_i) != c, via an auxiliary sum variable.
    """
    s = _aux_lin_sum(model, model.int_list_of(args[0]), model.var_list_of(args[1]))
    model.problem.add_propagator(ALG_NEQ_REIF, [model.var_index_of(args[3]), s, model.var_index_of(args[2])])


def _int_lin_ge(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_ge(a, x, c)`` as the linear inequality sum(a_i * x_i) >= c.
    """
    coeffs = model.int_list_of(args[0])
    variables = model.var_list_of(args[1])
    model.problem.add_propagator(ALG_LINEAR_GEQ_C, variables, coeffs + [model.const_of(args[2])])


def _int_lin_ge_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lin_ge_reif(a, x, c, r)`` as r <=> sum(a_i * x_i) >= c, i.e. r <=> c <= s, via an
    auxiliary sum variable.
    """
    s = _aux_lin_sum(model, model.int_list_of(args[0]), model.var_list_of(args[1]))
    model.problem.add_propagator(ALG_LEQ_C_REIF, [model.var_index_of(args[3]), model.var_index_of(args[2]), s], [0])


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


def _array_bool_or(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_bool_or(as, r)`` as r <=> (or of the booleans in as), i.e. r = max(as) over 0..1.
    """
    variables = model.var_list_of(args[0]) + [model.var_index_of(args[1])]
    model.problem.add_propagator(ALG_MAX_EQ, variables)


def _bool_or(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_or(a, b, r)`` as r <=> (a or b), i.e. r = max(a, b) over 0..1.
    """
    model.problem.add_propagator(
        ALG_MAX_EQ, [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
    )


def _bool_clause(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_clause(pos, neg)`` as the clause (or of pos) or (or of not neg), modelled as the linear
    inequality sum(pos) - sum(neg) >= 1 - len(neg).
    """
    pos = model.var_list_of(args[0])
    neg = model.var_list_of(args[1])
    coeffs = [1] * len(pos) + [-1] * len(neg)
    model.problem.add_propagator(ALG_LINEAR_GEQ_C, pos + neg, coeffs + [1 - len(neg)])


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


def _le_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_le_reif(x, y, r)`` / ``bool_le_reif(x, y, r)`` as r <=> x <= y.
    """
    r = model.var_index_of(args[2])
    model.problem.add_propagator(ALG_LEQ_C_REIF, [r, model.var_index_of(args[0]), model.var_index_of(args[1])], [0])


def _lt_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_lt_reif(x, y, r)`` / ``bool_lt_reif(x, y, r)`` as r <=> x <= y - 1.
    """
    r = model.var_index_of(args[2])
    model.problem.add_propagator(ALG_LEQ_C_REIF, [r, model.var_index_of(args[0]), model.var_index_of(args[1])], [-1])


def _ne_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_ne_reif(x, y, r)`` / ``bool_xor(x, y, r)`` as r <=> x != y.
    """
    r = model.var_index_of(args[2])
    model.problem.add_propagator(ALG_NEQ_REIF, [r, model.var_index_of(args[0]), model.var_index_of(args[1])])


def _int_ge(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_ge(x, y)`` as y <= x.
    """
    model.problem.add_propagator(ALG_LEQ_C, [model.var_index_of(args[1]), model.var_index_of(args[0])], [0])


def _int_gt(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_gt(x, y)`` as y <= x - 1.
    """
    model.problem.add_propagator(ALG_LEQ_C, [model.var_index_of(args[1]), model.var_index_of(args[0])], [-1])


def _ge_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_ge_reif(x, y, r)`` / ``bool_ge_reif(x, y, r)`` as r <=> y <= x.
    """
    r = model.var_index_of(args[2])
    model.problem.add_propagator(ALG_LEQ_C_REIF, [r, model.var_index_of(args[1]), model.var_index_of(args[0])], [0])


def _gt_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_gt_reif(x, y, r)`` / ``bool_gt_reif(x, y, r)`` as r <=> y <= x - 1.
    """
    r = model.var_index_of(args[2])
    model.problem.add_propagator(ALG_LEQ_C_REIF, [r, model.var_index_of(args[1]), model.var_index_of(args[0])], [-1])


def _bool_xor(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``bool_xor(a, b)`` as a != b and ``bool_xor(a, b, r)`` as the reification r <=> (a != b).
    """
    if len(args) == 3:
        _ne_reif(model, args)
    else:
        model.problem.add_propagator(ALG_NEQ, [model.var_index_of(args[0]), model.var_index_of(args[1])])


def _int_ne(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``int_ne(x, y)`` as x != y.
    """
    model.problem.add_propagator(ALG_NEQ, [model.var_index_of(args[0]), model.var_index_of(args[1])])


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
    Handles ``int_times(x, y, z)`` as x * y = z, using the constant-factor propagator when x or y is a
    constant and the general var * var propagator otherwise.
    """
    if _is_const(model, args[1]):
        model.problem.add_propagator(
            ALG_MUL_C_EQ, [model.var_index_of(args[0]), model.var_index_of(args[2])], [model.const_of(args[1])]
        )
    elif _is_const(model, args[0]):
        model.problem.add_propagator(
            ALG_MUL_C_EQ, [model.var_index_of(args[1]), model.var_index_of(args[2])], [model.const_of(args[0])]
        )
    else:
        model.problem.add_propagator(
            ALG_MUL_EQ, [model.var_index_of(args[0]), model.var_index_of(args[1]), model.var_index_of(args[2])]
        )


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


def _array_int_maximum(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_int_maximum(m, x)`` as m = max(x).
    """
    model.problem.add_propagator(ALG_MAX_EQ, model.var_list_of(args[1]) + [model.var_index_of(args[0])])


def _array_int_minimum(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``array_int_minimum(m, x)`` as m = min(x).
    """
    model.problem.add_propagator(ALG_MIN_EQ, model.var_list_of(args[1]) + [model.var_index_of(args[0])])


def _count_eq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``count_eq(x, y, c)`` as the number of x_i equal to y is c, supported only when y is a constant.
    """
    if not _is_const(model, args[1]):
        raise FznUnsupportedError("count_eq with a variable value is not supported")
    value = model.const_of(args[1])
    if _is_const(model, args[2]):
        model.problem.add_propagator(ALG_COUNT_EQ_C, model.var_list_of(args[0]), [value, model.const_of(args[2])])
    else:
        variables = model.var_list_of(args[0]) + [model.var_index_of(args[2])]
        model.problem.add_propagator(ALG_COUNT_EQ, variables, [value])


def _count_geq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``count_geq(x, y, c)`` as the number of x_i equal to y is at least c, supported only when y and c
    are constants.
    """
    if not _is_const(model, args[1]) or not _is_const(model, args[2]):
        raise FznUnsupportedError("count_geq with a variable value or count is not supported")
    model.problem.add_propagator(
        ALG_COUNT_GEQ_C, model.var_list_of(args[0]), [model.const_of(args[1]), model.const_of(args[2])]
    )


def _count_leq(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``count_leq(x, y, c)`` as the number of x_i equal to y is at most c, supported only when y and c
    are constants.
    """
    if not _is_const(model, args[1]) or not _is_const(model, args[2]):
        raise FznUnsupportedError("count_leq with a variable value or count is not supported")
    model.problem.add_propagator(
        ALG_COUNT_LEQ_C, model.var_list_of(args[0]), [model.const_of(args[1]), model.const_of(args[2])]
    )


def _nvalue(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``nvalue(n, x)`` as n being the number of distinct values taken by the array x.
    """
    variables = model.var_list_of(args[1]) + [model.var_index_of(args[0])]
    model.problem.add_propagator(ALG_NVALUE, variables)


def _all_different(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``all_different_int(x)``.
    """
    model.problem.add_propagator(ALG_ALLDIFFERENT, model.var_list_of(args[0]))


def _increasing(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``increasing_int(x)`` as x_i <= x_{i+1} for all i.
    """
    model.problem.add_propagator(ALG_INCREASING, model.var_list_of(args[0]))


def _strictly_increasing(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``strictly_increasing_int(x)`` as x_i < x_{i+1} for all i.
    """
    model.problem.add_propagator(ALG_STRICTLY_INCREASING, model.var_list_of(args[0]))


def _decreasing(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``decreasing_int(x)`` as x_i >= x_{i+1} for all i, i.e. increasing over the reversed list.
    """
    model.problem.add_propagator(ALG_INCREASING, model.var_list_of(args[0])[::-1])


def _strictly_decreasing(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``strictly_decreasing_int(x)`` as x_i > x_{i+1} for all i, i.e. strictly increasing over the
    reversed list.
    """
    model.problem.add_propagator(ALG_STRICTLY_INCREASING, model.var_list_of(args[0])[::-1])


def _zero_based(model: "FznModel", variables: List[int], n: int) -> List[int]:
    """
    Returns auxiliary 0-based copies (value - 1) of 1-based FlatZinc variables, for propagators that work
    on a 0-based index/value representation.
    """
    shifted = []
    for v in variables:
        v0 = model.problem.add_variable((0, n - 1))
        model.problem.add_propagator(ALG_ADD_C_EQ, [v, v0], [-1])
        shifted.append(v0)
    return shifted


def _circuit(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``circuit(x)``: the 1-based successor array x forms a single Hamiltonian circuit. NuCS uses a
    0-based successor representation, so each x[i] is shifted by -1 into an auxiliary variable, on which a
    permutation (alldifferent) plus a no-sub-cycle constraint together enforce the circuit.
    """
    succ = model.var_list_of(args[0])
    shifted = _zero_based(model, succ, len(succ))
    model.problem.add_propagator(ALG_ALLDIFFERENT, shifted)
    model.problem.add_propagator(ALG_NO_SUB_CYCLE, shifted)


def _subcircuit(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``subcircuit(x)``: the 1-based successor array x forms a sub-circuit -- the nodes with x[i] != i
    form a single circuit and the rest are self-loops (excluded). Like ``circuit``, NuCS works on a 0-based
    representation, so x is shifted by -1; a permutation (alldifferent) plus a self-loop-aware sub-circuit
    constraint enforce it.
    """
    succ = model.var_list_of(args[0])
    shifted = _zero_based(model, succ, len(succ))
    model.problem.add_propagator(ALG_ALLDIFFERENT, shifted)
    model.problem.add_propagator(ALG_SUBCIRCUIT, shifted)


def _inverse(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``inverse(x, y)`` as the channeling x[i] = j <=> y[j] = i between two 1-based arrays. The
    channeling propagator works on a 0-based representation, so both arrays are shifted by -1; an
    alldifferent on each array is added for stronger (Hall) pruning than the channeling alone provides.
    """
    x = model.var_list_of(args[0])
    y = model.var_list_of(args[1])
    n = len(x)
    model.problem.add_propagator(ALG_ALLDIFFERENT, x)
    model.problem.add_propagator(ALG_ALLDIFFERENT, y)
    model.problem.add_propagator(ALG_INVERSE, _zero_based(model, x, n) + _zero_based(model, y, n))


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


def _set_in(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``set_in(x, S)`` as x being a member of the set S (a ``{..}`` literal or a ``lo..hi`` range).
    """
    model.problem.add_propagator(ALG_MEMBER, [model.var_index_of(args[0])], model.set_values_of(args[1]))


def _set_in_reif(model: "FznModel", args: List[Term]) -> None:
    """
    Handles ``set_in_reif(x, S, b)`` as b <=> (x in S), where S is a constant set (a ``{..}`` literal or a
    ``lo..hi`` range). A contiguous range reifies to lo <= x <= hi; a non-contiguous set to a disjunction of
    equalities.
    """
    x = model.var_index_of(args[0])
    b = model.var_index_of(args[2])
    values = model.set_values_of(args[1])
    if not values:  # x is never in the empty set, so b is false
        model.problem.add_propagator(ALG_LEQ_C, [b, model.var_index_of(0)], [0])
        return
    if len(values) == 1:  # b <=> x == v
        model.problem.add_propagator(ALG_EQ_C_REIF, [b, x], [values[0]])
        return
    lo, hi = values[0], values[-1]
    if hi - lo + 1 == len(values):  # contiguous range: b <=> (lo <= x) and (x <= hi)
        b_ge = model.problem.add_variable((0, 1))
        b_le = model.problem.add_variable((0, 1))
        model.problem.add_propagator(ALG_LEQ_C_REIF, [b_ge, model.var_index_of(lo), x], [0])
        model.problem.add_propagator(ALG_LEQ_C_REIF, [b_le, x, model.var_index_of(hi)], [0])
        model.problem.add_propagator(ALG_AND_EQ, [b_ge, b_le, b])
        return
    reifs = []  # non-contiguous set: b <=> or_i (x == v_i)
    for value in values:
        reif = model.problem.add_variable((0, 1))
        model.problem.add_propagator(ALG_EQ_C_REIF, [reif, x], [value])
        reifs.append(reif)
    model.problem.add_propagator(ALG_MAX_EQ, reifs + [b])


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
    index0 = model.problem.add_variable((0, len(array) - 1))
    model.problem.add_propagator(ALG_ADD_C_EQ, [index, index0], [-1])
    if _is_const(model, args[2]):
        model.problem.add_propagator(ALG_ELEMENT_L_EQ_C, array + [index0], [model.const_of(args[2])])
    else:
        model.problem.add_propagator(ALG_ELEMENT_L_EQ, array + [index0, model.var_index_of(args[2])])


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
    "array_bool_element": _array_int_element,
    "array_bool_or": _array_bool_or,
    "array_int_element": _array_int_element,
    "array_int_maximum": _array_int_maximum,
    "array_int_minimum": _array_int_minimum,
    "array_var_bool_element": _array_var_int_element,
    "array_var_int_element": _array_var_int_element,
    "bool2int": _bool2int,
    "bool_and": _bool_and,
    "bool_clause": _bool_clause,
    "bool_eq": _bool_eq,
    "bool_eq_reif": _int_eq_reif,
    "bool_ge_reif": _ge_reif,
    "bool_gt_reif": _gt_reif,
    "bool_le": _bool_le,
    "bool_le_reif": _le_reif,
    "bool_lin_eq": _int_lin_eq,
    "bool_lin_le": _int_lin_le,
    "bool_lt": _int_lt,
    "bool_lt_reif": _lt_reif,
    "bool_not": _bool_not,
    "bool_or": _bool_or,
    "bool_xor": _bool_xor,
    "circuit": _circuit,
    "count_eq": _count_eq,
    "count_geq": _count_geq,
    "count_leq": _count_leq,
    "decreasing_int": _decreasing,
    "fzn_all_different_int": _all_different,
    "fzn_circuit": _circuit,
    "fzn_count_eq": _count_eq,
    "fzn_count_geq": _count_geq,
    "fzn_count_leq": _count_leq,
    "fzn_decreasing_int": _decreasing,
    "fzn_global_cardinality_low_up": _global_cardinality_low_up,
    "fzn_increasing_int": _increasing,
    "fzn_inverse": _inverse,
    "fzn_lex_lesseq_int": _lex_lesseq,
    "fzn_nvalue": _nvalue,
    "fzn_strictly_decreasing_int": _strictly_decreasing,
    "fzn_strictly_increasing_int": _strictly_increasing,
    "fzn_subcircuit": _subcircuit,
    "global_cardinality_low_up": _global_cardinality_low_up,
    "increasing_int": _increasing,
    "int_abs": _int_abs,
    "int_eq": _int_eq,
    "int_eq_reif": _int_eq_reif,
    "int_ge": _int_ge,
    "int_ge_reif": _ge_reif,
    "int_gt": _int_gt,
    "int_gt_reif": _gt_reif,
    "int_le": _int_le,
    "int_le_reif": _le_reif,
    "int_lin_eq": _int_lin_eq,
    "int_lin_eq_reif": _int_lin_eq_reif,
    "int_lin_ge": _int_lin_ge,
    "int_lin_ge_reif": _int_lin_ge_reif,
    "int_lin_le": _int_lin_le,
    "int_lin_le_reif": _int_lin_le_reif,
    "int_lin_ne": _int_lin_ne,
    "int_lin_ne_reif": _int_lin_ne_reif,
    "int_lt": _int_lt,
    "int_lt_reif": _lt_reif,
    "int_max": _int_max,
    "int_min": _int_min,
    "int_ne": _int_ne,
    "int_ne_reif": _ne_reif,
    "inverse": _inverse,
    "int_plus": _int_plus,
    "int_times": _int_times,
    "lex_lesseq_int": _lex_lesseq,
    "nucs_table_int": _table_int,
    "nvalue": _nvalue,
    "set_in": _set_in,
    "set_in_reif": _set_in_reif,
    "strictly_decreasing_int": _strictly_decreasing,
    "strictly_increasing_int": _strictly_increasing,
    "subcircuit": _subcircuit,
}
