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
import io
import json
from typing import Optional

import pytest

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run, search_heuristics
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
)
from nucs.propagators.propagators import (
    ALG_ADD_C_EQ,
    ALG_ALLDIFFERENT,
    ALG_COUNT_EQ,
    ALG_COUNT_EQ_C,
    ALG_COUNT_GEQ_C,
    ALG_COUNT_LEQ_C,
    ALG_ELEMENT_L_EQ,
    ALG_ELEMENT_L_EQ_C,
    ALG_INCREASING,
    ALG_LEQ_C,
    ALG_LEXLEQ,
    ALG_LINEAR_EQ_C,
    ALG_MOD_C_EQ,
    ALG_MOD_EQ,
    ALG_NO_SUB_CYCLE,
    ALG_NVALUE,
    ALG_STRICTLY_INCREASING,
    ALG_SUBCIRCUIT,
    ALG_SUM_EQ_C,
    ALG_SUM_GEQ_C,
    ALG_SUM_LEQ_C,
    ALG_VALUE_PRECEDE,
)


def solve_fzn(
    fzn: str,
    all_solutions: bool = False,
    num_solutions: Optional[int] = None,
    output_mode: str = "item",
    output_objective: bool = False,
) -> str:
    """Builds, solves and returns the FlatZinc solution stream as text."""
    out = io.StringIO()
    run(
        build_model(parse(fzn)),
        out,
        io.StringIO(),
        all_solutions=all_solutions,
        num_solutions=num_solutions,
        output_mode=output_mode,
        output_objective=output_objective,
    )
    return out.getvalue()


class TestBuiltins:
    def test_mapping_propagators(self) -> None:
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\n"
                "constraint int_lin_eq([2, 1], [x, y], 5);\n"  # non-unit coefficients -> general propagator
                "constraint int_le(x, y);\n"
                "constraint all_different_int([x, y]);\n"
                "solve satisfy;"
            )
        )
        algorithms = [prop[1] for prop in model.problem.propagators]
        assert algorithms == [ALG_LINEAR_EQ_C, ALG_LEQ_C, ALG_ALLDIFFERENT]
        # int_lin_eq([2,1], [x,y], 5) -> linear_eq_c with params coeffs + [c]
        assert model.problem.propagators[0][2] == [2, 1, 5]

    def test_unit_coefficient_linear_routes_to_sum(self) -> None:
        # all-1 coefficients use the cheaper plain-sum propagators (param is just [c], no coefficients)
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\n"
                "constraint int_lin_eq([1, 1], [x, y], 5);\n"
                "constraint int_lin_le([1, 1], [x, y], 7);\n"
                "constraint int_lin_ge([1, 1], [x, y], 3);\n"
                "solve satisfy;"
            )
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_SUM_EQ_C, ALG_SUM_LEQ_C, ALG_SUM_GEQ_C]
        assert [prop[2] for prop in model.problem.propagators] == [[5], [7], [3]]

    def test_negative_unit_coefficient_linear_routes_to_sum(self) -> None:
        # all-(-1) coefficients negate the constant and flip le<->ge: sum(-x) <= c  <=>  sum(x) >= -c
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\n"
                "constraint int_lin_eq([-1, -1], [x, y], -5);\n"
                "constraint int_lin_le([-1, -1], [x, y], -3);\n"
                "solve satisfy;"
            )
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_SUM_EQ_C, ALG_SUM_GEQ_C]
        assert [prop[2] for prop in model.problem.propagators] == [[5], [3]]

    def test_unit_coefficient_sum_solves_correctly(self) -> None:
        # x + y = 5, x + y <= 7 (redundant), x <= y: same solutions as the general propagator would give
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..9: y :: output_var;\n"
            "constraint int_lin_eq([1, 1], [x, y], 5);\nconstraint int_le(x, y);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3  # (0,5), (1,4), (2,3)
        assert "x = 0;\ny = 5;" in out and "x = 2;\ny = 3;" in out and "x = 3;" not in out

    def test_bool_lin_eq_variable_result(self) -> None:
        # bool_lin_eq's result is a variable (unlike int_lin_eq's constant): sum of true booleans = s
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: c :: output_var;\n"
            "var 0..3: s :: output_var;\n"
            "constraint bool_lin_eq([1, 1, 1], [a, b, c], s);\nconstraint int_eq(s, 2);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3  # exactly two of the three booleans are true
        assert "s = 2;" in out and "s = 3;" not in out

    def test_bool_lin_eq_non_unit_coefficients(self) -> None:
        # non-unit coefficients route through the general linear propagator: 2a + 3b = 3 forces a=0, b=1
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar 0..5: s :: output_var;\n"
            "constraint bool_lin_eq([2, 3], [a, b], s);\nconstraint int_eq(s, 3);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 1
        assert "a = false;" in out and "b = true;" in out

    def test_all_different_plus_linear_satisfy(self) -> None:
        out = solve_fzn(
            "array [1..4] of int: c = [1, 1, 1, 1];\n"
            "var 1..4: a :: output_var;\nvar 1..4: b :: output_var;\n"
            "var 1..4: d :: output_var;\nvar 1..4: e :: output_var;\n"
            "array [1..4] of var int: q = [a, b, d, e];\n"
            "constraint fzn_all_different_int(q);\n"
            "constraint int_lin_eq(c, [a, b, d, e], 10);\n"
            "solve satisfy;"
        )
        assert "a = 1;" in out and "b = 2;" in out and "d = 3;" in out and "e = 4;" in out
        assert out.strip().endswith("----------")

    def test_element_one_based_offset(self) -> None:
        out = solve_fzn(
            "array [1..3] of int: A = [10, 20, 30];\n"
            "var 2..2: i :: output_var;\nvar 10..30: v :: output_var;\n"
            "constraint array_int_element(i, A, v);\n"
            "solve satisfy;"
        )
        assert "v = 20;" in out  # A[2] (1-based) == 20

    def test_var_element(self) -> None:
        out = solve_fzn(
            "var 7..7: w0;\nvar 8..8: w1;\nvar 9..9: w2;\n"
            "array [1..3] of var int: X = [w0, w1, w2];\n"
            "var 3..3: i :: output_var;\nvar 0..100: v :: output_var;\n"
            "constraint array_var_int_element(i, X, v);\n"
            "solve satisfy;"
        )
        assert "v = 9;" in out  # X[3] (1-based) == 9

    def test_var_element_variable_value_uses_element_l_eq(self) -> None:
        model = build_model(
            parse(
                "var 0..9: a;\nvar 0..9: b;\nvar 1..2: i;\nvar 0..9: v;\n"
                "array [1..2] of var int: X = [a, b];\n"
                "constraint array_var_int_element(i, X, v);\nsolve satisfy;"
            )
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_ADD_C_EQ, ALG_ELEMENT_L_EQ]

    def test_var_element_constant_value_uses_element_l_eq_c(self) -> None:
        # a constant value routes to the specialized element_l_eq_c propagator, no auxiliary value variable
        model = build_model(
            parse(
                "var 0..9: a;\nvar 0..9: b;\nvar 1..2: i;\n"
                "array [1..2] of var int: X = [a, b];\n"
                "constraint array_var_int_element(i, X, 7);\nsolve satisfy;"
            )
        )
        algorithms = [prop[1] for prop in model.problem.propagators]
        assert algorithms == [ALG_ADD_C_EQ, ALG_ELEMENT_L_EQ_C]
        assert model.problem.propagators[1][2] == [7]  # params = [value]

    def test_var_element_constant_value_solves(self) -> None:
        # X[i] = 8 with X = [7, 8, 9] selects index 2 (1-based)
        out = solve_fzn(
            "var 7..7: w0;\nvar 8..8: w1;\nvar 9..9: w2;\n"
            "array [1..3] of var int: X = [w0, w1, w2];\n"
            "var 1..3: i :: output_var;\n"
            "constraint array_var_int_element(i, X, 8);\n"
            "solve satisfy;"
        )
        assert "i = 2;" in out

    def test_gcc_permutation(self) -> None:
        out = solve_fzn(
            "var 0..2: a :: output_var;\nvar 0..2: b :: output_var;\nvar 0..2: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "array [1..3] of int: cover = [0, 1, 2];\n"
            "array [1..3] of int: lb = [1, 1, 1];\n"
            "array [1..3] of int: ub = [1, 1, 1];\n"
            "constraint fzn_global_cardinality_low_up(x, cover, lb, ub);\n"
            "constraint int_lt(a, b);\nconstraint int_lt(b, c);\n"
            "solve satisfy;"
        )
        assert "a = 0;" in out and "b = 1;" in out and "c = 2;" in out

    def test_gcc_values_outside_cover_are_unconstrained(self) -> None:
        # alldifferent_except_0 lowers to a gcc whose cover is only the non-zero values; values outside the
        # cover (here 0) must stay unconstrained and may repeat. Forcing every variable to 0 must remain
        # satisfiable even though 0 is not in the cover (regression for the spurious-UNSAT GCC bug).
        out = solve_fzn(
            "var 0..3: a :: output_var;\nvar 0..3: b :: output_var;\nvar 0..3: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "array [1..2] of int: cover = [1, 2];\n"  # only 1 and 2 are covered; 0 and 3 are free
            "array [1..2] of int: lb = [0, 0];\n"
            "array [1..2] of int: ub = [1, 1];\n"
            "constraint fzn_global_cardinality_low_up(x, cover, lb, ub);\n"
            "constraint int_eq(a, 0);\nconstraint int_eq(b, 0);\nconstraint int_eq(c, 0);\n"
            "solve satisfy;"
        )
        assert "a = 0;" in out and "b = 0;" in out and "c = 0;" in out

    def test_disjunctive(self) -> None:
        # two length-3 tasks on a unary resource: s[0] is pinned to [0,2], forcing s[1] to start at/after 3
        out = solve_fzn(
            "array [1..2] of var 0..5: s :: output_array([1..2]);\n"
            "array [1..2] of int: d = [3, 3];\n"
            "constraint int_le(s[1], 2);\n"
            "constraint nucs_disjunctive(s, d);\n"
            "solve satisfy;",
        )
        # s[1] (0-based variable 0) <= 2, so s[2] >= 3; the first solution sets them to their minima
        assert "s = array1d(1..2, [0, 3]);" in out

    def test_diffn(self) -> None:
        # rectangle 0 is a 2x2 square pinned at the origin; rectangle 1 (also 2x2) is pinned to overlap it
        # vertically (y in [0,1]), so it is forced to the right: x[1] >= 2
        out = solve_fzn(
            "array [1..2] of var 0..5: x :: output_array([1..2]);\n"
            "array [1..2] of var 0..5: y;\n"
            "array [1..2] of int: dx = [2, 2];\narray [1..2] of int: dy = [2, 2];\n"
            "constraint int_eq(x[1], 0);\nconstraint int_eq(y[1], 0);\nconstraint int_le(y[2], 1);\n"
            "constraint nucs_diffn(x, y, dx, dy);\nsolve satisfy;",
        )
        # x[1] (0-based variable 0) = 0; first solution puts x[2] at its forced minimum 2
        assert "x = array1d(1..2, [0, 2]);" in out

    def test_diffn_inconsistent(self) -> None:
        # two unit squares both pinned to the same cell must overlap
        out = solve_fzn(
            "array [1..2] of var 0..0: x;\narray [1..2] of var 0..0: y;\n"
            "array [1..2] of int: dx = [1, 1];\narray [1..2] of int: dy = [1, 1];\n"
            "constraint nucs_diffn(x, y, dx, dy);\nsolve satisfy;",
        )
        assert "UNSATISFIABLE" in out

    def test_disjunctive_inconsistent(self) -> None:
        # three length-2 tasks cannot all fit, non-overlapping, into the window [0, 5]
        out = solve_fzn(
            "array [1..3] of var 0..3: s;\narray [1..3] of int: d = [2, 2, 2];\n"
            "constraint nucs_disjunctive(s, d);\nsolve satisfy;",
        )
        assert "UNSATISFIABLE" in out

    def test_bool_builtins_and_bool_output(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, true);\n"  # a = true
            "constraint bool_not(a, b);\n"  # b = not a = false
            "constraint bool_le(b, a);\n"  # false <= true, consistent
            "array [1..2] of var bool: ps = [a, b];\n"
            "constraint array_bool_and(ps, r);\n"  # r = a /\\ b = false
            "solve satisfy;"
        )
        # booleans must be rendered as true/false, not 1/0
        assert "a = true;" in out
        assert "b = false;" in out
        assert "r = false;" in out

    def test_table(self) -> None:
        out = solve_fzn(
            "var 1..3: a :: output_var;\nvar 1..3: b :: output_var;\n"
            "array [1..2] of var int: x = [a, b];\n"
            "array [1..6] of int: t = [1, 2, 2, 3, 3, 1];\n"  # rows (1,2) (2,3) (3,1), flattened row-major
            "constraint nucs_table_int(x, t);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3  # exactly the three table rows
        assert "a = 1;\nb = 2;" in out and "a = 2;\nb = 3;" in out and "a = 3;\nb = 1;" in out

    def test_int_ne(self) -> None:
        out = solve_fzn(
            "var 1..2: x :: output_var;\nvar 1..2: y :: output_var;\n"
            "constraint int_ne(x, y);\nconstraint int_lt(x, y);\n"
            "solve satisfy;"
        )
        assert "x = 1;" in out and "y = 2;" in out

    def test_int_lin_ne(self) -> None:
        out = solve_fzn(
            "var 1..2: x :: output_var;\nvar 1..2: y :: output_var;\n"
            "constraint int_lin_ne([1, 1], [x, y], 3);\nconstraint int_lt(x, y);\n"
            "solve satisfy;"
        )
        # x < y forces (1, 2) which sums to 3, excluded by the disequality -> unsatisfiable
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_set_in(self) -> None:
        out = solve_fzn(
            "var 0..10: x :: output_var;\nconstraint set_in(x, {1, 3, 5});\nconstraint int_lt(x, 4);\nsolve satisfy;",
            all_solutions=True,
        )
        # x in {1,3,5} and x < 4 -> {1, 3}
        assert out.count("----------") == 2
        assert "x = 1;" in out and "x = 3;" in out

    def test_non_contiguous_var_domain(self) -> None:
        out = solve_fzn(
            "var {1, 3, 5}: x :: output_var;\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert "x = 1;" in out and "x = 3;" in out and "x = 5;" in out

    def test_set_in_reif_range_true(self) -> None:
        # b is true so x must be in the contiguous range 1..2
        out = solve_fzn(
            "var 0..4: x :: output_var;\nconstraint set_in_reif(x, 1..2, true);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 2
        assert "x = 1;" in out and "x = 2;" in out

    def test_set_in_reif_range_false(self) -> None:
        # b is false so x must be outside 1..2
        out = solve_fzn(
            "var 0..4: x :: output_var;\nconstraint set_in_reif(x, 1..2, false);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert "x = 0;" in out and "x = 3;" in out and "x = 4;" in out
        assert "x = 1;" not in out and "x = 2;" not in out

    def test_set_in_reif_non_contiguous(self) -> None:
        # b is true so x must be one of the (non-contiguous) literal values
        out = solve_fzn(
            "var 0..4: x :: output_var;\nconstraint set_in_reif(x, {1, 3}, true);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 2
        assert "x = 1;" in out and "x = 3;" in out

    def test_set_in_reif_derives_reif(self) -> None:
        # x is fixed to 2 which is in 1..3, so b is forced true
        out = solve_fzn("var 2..2: x;\nvar bool: b :: output_var;\nconstraint set_in_reif(x, 1..3, b);\nsolve satisfy;")
        assert "b = true;" in out

    def test_set_in_reif_used_for_counting(self) -> None:
        # set_in_reif feeding a linear sum on the reif bools is exactly how `among` is decomposed:
        # forcing the count of members to 2 makes both x and y land in 1..2
        out = solve_fzn(
            "var 0..3: x :: output_var;\nvar 0..3: y :: output_var;\nvar bool: bx;\nvar bool: by;\n"
            "constraint set_in_reif(x, 1..2, bx);\nconstraint set_in_reif(y, 1..2, by);\n"
            "constraint bool_lin_eq([1, 1], [bx, by], 2);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 4  # x, y each independently in {1, 2}
        assert "x = 0;" not in out and "x = 3;" not in out

    def test_int_le_reif(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_le_reif(x, 2, r);\nconstraint bool_eq(r, false);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # r is false so x <= 2 must not hold -> x in {3, 4, 5}
        assert out.count("----------") == 3
        assert "x = 3;" in out and "x = 5;" in out
        assert "x = 2;" not in out

    def test_int_lt_reif(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_lt_reif(x, 3, r);\nconstraint bool_eq(r, true);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # r is true so x < 3 -> x in {0, 1, 2}
        assert out.count("----------") == 3
        assert "x = 0;" in out and "x = 2;" in out
        assert "x = 3;" not in out

    def test_bool_le_reif_detects_truth_value(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, true);\nconstraint bool_eq(b, false);\n"
            "constraint bool_le_reif(a, b, r);\n"  # r <=> (true <= false) -> r = false
            "solve satisfy;"
        )
        assert "r = false;" in out

    def test_int_ne_reif(self) -> None:
        out = solve_fzn(
            "var 0..3: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_ne_reif(x, 2, r);\nconstraint bool_eq(r, false);\n"
            "solve satisfy;"
        )
        # r is false so x != 2 must not hold -> x = 2
        assert "x = 2;" in out

    def test_bool_xor_reif(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, true);\nconstraint bool_eq(b, false);\n"
            "constraint bool_xor(a, b, r);\n"  # r <=> (true xor false) -> r = true
            "solve satisfy;"
        )
        assert "r = true;" in out

    def test_bool_xor_binary(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\n"
            "constraint bool_xor(a, b);\nconstraint bool_eq(a, true);\n"  # a xor b, a = true -> b = false
            "solve satisfy;"
        )
        assert "a = true;" in out and "b = false;" in out

    def test_bool_or(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, true);\nconstraint bool_eq(b, false);\n"
            "constraint bool_or(a, b, r);\n"  # r <=> (true or false) -> r = true
            "solve satisfy;"
        )
        assert "r = true;" in out

    def test_array_bool_or(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, false);\nconstraint bool_eq(b, false);\n"
            "array [1..2] of var bool: ps = [a, b];\n"
            "constraint array_bool_or(ps, r);\n"  # r <=> (false or false) -> r = false
            "solve satisfy;"
        )
        assert "r = false;" in out

    def test_bool_clause(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\n"
            "constraint bool_eq(a, false);\n"
            "constraint bool_clause([a], [b]);\n"  # a or not b, with a = false -> b = false
            "solve satisfy;"
        )
        assert "a = false;" in out and "b = false;" in out

    def test_bool_eq_reif(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\nvar bool: r :: output_var;\n"
            "constraint bool_eq(a, true);\nconstraint bool_eq(b, false);\n"
            "constraint bool_eq_reif(a, b, r);\n"  # r <=> (true = false) -> r = false
            "solve satisfy;"
        )
        assert "r = false;" in out

    def test_bool_lin_le(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\n"
            "constraint bool_eq(a, true);\n"
            "constraint bool_lin_le([1, 1], [a, b], 1);\n"  # a + b <= 1, a = true -> b = false
            "solve satisfy;"
        )
        assert "a = true;" in out and "b = false;" in out

    def test_array_bool_element(self) -> None:
        out = solve_fzn(
            "array [1..3] of bool: A = [true, false, true];\n"
            "var 2..2: i :: output_var;\nvar bool: c :: output_var;\n"
            "constraint array_bool_element(i, A, c);\n"  # c = A[2] (1-based) = false
            "solve satisfy;"
        )
        assert "c = false;" in out

    def test_int_lin_eq_reif(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar 0..5: y :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_lin_eq_reif([1, 1], [x, y], 4, r);\nconstraint bool_eq(r, true);\n"
            "constraint int_eq(x, 1);\n"
            "solve satisfy;"
        )
        # r true so x + y = 4, x = 1 -> y = 3
        assert "y = 3;" in out

    def test_int_lin_le_reif(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_lin_le_reif([1], [x], 2, r);\nconstraint bool_eq(r, false);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # r false so NOT(x <= 2) -> x in {3, 4, 5}
        assert out.count("----------") == 3
        assert "x = 3;" in out and "x = 5;" in out
        assert "x = 2;" not in out

    def test_int_lin_ne_reif(self) -> None:
        out = solve_fzn(
            "var 0..3: x :: output_var;\nvar 0..3: y :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_lin_ne_reif([1, 1], [x, y], 2, r);\nconstraint bool_eq(r, false);\n"
            "constraint int_eq(x, 0);\n"
            "solve satisfy;"
        )
        # r false so x + y = 2 must hold, x = 0 -> y = 2
        assert "y = 2;" in out

    def test_int_lin_le_reif_binary_difference(self) -> None:
        # the [1,-1]/[-1,1] difference pattern is posted directly as r <=> x <= y + c (no aux variable);
        # check both the reif direction (deriving r) and the filtering direction (deriving x from r)
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_eq(x, 4);\nvar 0..9: y;\nconstraint int_eq(y, 2);\n"
            "constraint int_lin_le_reif([1, -1], [x, y], 1, r);\n"  # 4 - 2 = 2 <= 1 ? no -> r false
            "solve satisfy;"
        )
        assert "r = false;" in out
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..9: y :: output_var;\nvar bool: r;\n"
            "constraint int_eq(y, 3);\nconstraint bool_eq(r, true);\n"
            "constraint int_lin_le_reif([-1, 1], [x, y], 1, r);\n"  # r true: y - x <= 1 -> x >= 2
            "solve satisfy;",
            all_solutions=True,
        )
        # x in {2,...,9}; never below 2
        assert "x = 2;" in out and "x = 1;" not in out and "x = 0;" not in out

    def test_int_lin_ge_reif_binary_difference(self) -> None:
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_eq(x, 4);\nvar 0..9: y;\nconstraint int_eq(y, 2);\n"
            "constraint int_lin_ge_reif([1, -1], [x, y], 3, r);\n"  # 4 - 2 = 2 >= 3 ? no -> r false
            "solve satisfy;"
        )
        assert "r = false;" in out
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..9: y;\nvar bool: r;\n"
            "constraint int_eq(y, 2);\nconstraint bool_eq(r, true);\n"
            "constraint int_lin_ge_reif([1, -1], [x, y], 3, r);\n"  # r true: x - 2 >= 3 -> x >= 5
            "solve satisfy;",
            all_solutions=True,
        )
        assert "x = 5;" in out and "x = 4;" not in out

    def test_array_int_maximum_minimum(self) -> None:
        out = solve_fzn(
            "var 0..9: a;\nvar 0..9: b;\nvar 0..9: c;\n"
            "var 0..9: lo :: output_var;\nvar 0..9: hi :: output_var;\n"
            "constraint int_eq(a, 3);\nconstraint int_eq(b, 7);\nconstraint int_eq(c, 5);\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint array_int_maximum(hi, x);\nconstraint array_int_minimum(lo, x);\n"
            "solve satisfy;"
        )
        assert "hi = 7;" in out and "lo = 3;" in out

    def test_int_ge_gt(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\n"
            "constraint int_ge(x, 3);\nconstraint int_gt(5, x);\n"  # x >= 3 and 5 > x -> x in {3, 4}
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 2
        assert "x = 3;" in out and "x = 4;" in out

    def test_int_ge_reif(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar bool: r :: output_var;\n"
            "constraint int_ge_reif(x, 3, r);\nconstraint bool_eq(r, true);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # r true so x >= 3 -> x in {3, 4, 5}
        assert out.count("----------") == 3
        assert "x = 3;" in out and "x = 5;" in out
        assert "x = 2;" not in out

    def test_int_lin_ge(self) -> None:
        out = solve_fzn(
            "var 0..5: x :: output_var;\nvar 0..5: y :: output_var;\n"
            "constraint int_lin_ge([1, 1], [x, y], 9);\nconstraint int_eq(x, 5);\n"  # x + y >= 9, x = 5 -> y >= 4
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 2  # y in {4, 5}
        assert "y = 4;" in out and "y = 5;" in out

    def test_bool_lt(self) -> None:
        out = solve_fzn(
            "var bool: a :: output_var;\nvar bool: b :: output_var;\n"
            "constraint bool_lt(a, b);\n"  # a < b -> a = false, b = true
            "solve satisfy;"
        )
        assert "a = false;" in out and "b = true;" in out

    def test_count_eq(self) -> None:
        out = solve_fzn(
            "var 1..3: a;\nvar 1..3: b;\nvar 1..3: c;\nvar 0..3: n :: output_var;\n"
            "constraint int_eq(a, 2);\nconstraint int_eq(b, 2);\nconstraint int_eq(c, 1);\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint count_eq(x, 2, n);\n"  # number of 2s among [2, 2, 1] -> 2
            "solve satisfy;"
        )
        assert "n = 2;" in out

    def test_count_eq_variable_value_unsupported(self) -> None:
        with pytest.raises(FznUnsupportedError):
            build_model(
                parse(
                    "var 1..3: a;\nvar 1..3: y;\nvar 0..1: n;\n"
                    "array [1..1] of var int: x = [a];\n"
                    "constraint count_eq(x, y, n);\nsolve satisfy;"
                )
            )

    def test_count_eq_constant_count_uses_count_eq_c(self) -> None:
        # a constant count routes to the specialized count_eq_c propagator, no auxiliary counter variable
        model = build_model(parse("array [1..3] of var 1..3: x;\nconstraint count_eq(x, 2, 2);\nsolve satisfy;"))
        assert [prop[1] for prop in model.problem.propagators] == [ALG_COUNT_EQ_C]
        assert model.problem.propagators[0][2] == [2, 2]  # params = [value, count]

    def test_count_eq_variable_count_uses_count_eq(self) -> None:
        # a variable count still uses the variable-counter propagator
        model = build_model(
            parse("var 0..3: n;\narray [1..3] of var 1..3: x;\nconstraint count_eq(x, 2, n);\nsolve satisfy;")
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_COUNT_EQ]

    def test_count_geq(self) -> None:
        # at least 2 of the three vars equal 2; fixing two to 2 and one to 1 satisfies it
        out = solve_fzn(
            "var 1..3: a :: output_var;\nvar 1..3: b :: output_var;\nvar 1..3: c :: output_var;\n"
            "constraint int_eq(c, 1);\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint count_geq(x, 2, 2);\n"
            "solve satisfy;"
        )
        assert "a = 2;" in out and "b = 2;" in out and "c = 1;" in out

    def test_count_geq_maps_to_count_geq_c(self) -> None:
        model = build_model(parse("array [1..3] of var 1..3: x;\nconstraint count_geq(x, 2, 2);\nsolve satisfy;"))
        assert [prop[1] for prop in model.problem.propagators] == [ALG_COUNT_GEQ_C]
        assert model.problem.propagators[0][2] == [2, 2]

    def test_count_leq_too_many_unsatisfiable(self) -> None:
        # at most 1 var may equal 2, but all three are forced to 2 -> inconsistent
        out = solve_fzn(
            "var 2..2: a;\nvar 2..2: b;\nvar 2..2: c;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint count_leq(x, 2, 1);\n"
            "solve satisfy;"
        )
        assert "----------" not in out

    def test_count_leq_maps_to_count_leq_c(self) -> None:
        model = build_model(parse("array [1..3] of var 1..3: x;\nconstraint count_leq(x, 2, 1);\nsolve satisfy;"))
        assert [prop[1] for prop in model.problem.propagators] == [ALG_COUNT_LEQ_C]
        assert model.problem.propagators[0][2] == [2, 1]

    def test_count_geq_variable_count_unsupported(self) -> None:
        with pytest.raises(FznUnsupportedError):
            build_model(
                parse("var 0..3: n;\narray [1..3] of var 1..3: x;\nconstraint count_geq(x, 2, n);\nsolve satisfy;")
            )

    def test_nvalue_maps_to_nvalue(self) -> None:
        # n is declared first (index 0), a/b/c next; nvalue puts the x variables first and the count last
        model = build_model(
            parse(
                "var 0..3: n;\nvar 0..2: a;\nvar 0..2: b;\nvar 0..2: c;\n"
                "array [1..3] of var int: x = [a, b, c];\n"
                "constraint fzn_nvalue(n, x);\nsolve satisfy;"
            )
        )
        (propagator,) = model.problem.propagators
        assert propagator[1] == ALG_NVALUE
        assert list(propagator[0]) == [1, 2, 3, 0]

    def test_nvalue_counts_distinct(self) -> None:
        out = solve_fzn(
            "var 0..3: n :: output_var;\nvar 1..1: a;\nvar 1..1: b;\nvar 2..2: c;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_nvalue(n, x);\nsolve satisfy;"
        )
        assert "n = 2;" in out  # {1, 1, 2} -> 2 distinct values

    def test_nvalue_one_forces_all_equal(self) -> None:
        # n = 1 forces a single distinct value: the intersection of [0,5], [2,3], [3,9] is {3}
        out = solve_fzn(
            "var 1..1: n;\nvar 0..5: a :: output_var;\nvar 2..3: b :: output_var;\nvar 3..9: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_nvalue(n, x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 1
        assert "a = 3;" in out and "b = 3;" in out and "c = 3;" in out

    def test_value_precede_maps_to_value_precede(self) -> None:
        model = build_model(
            parse("array [1..3] of var 0..3: x;\nconstraint fzn_value_precede_int(1, 2, x);\nsolve satisfy;")
        )
        (propagator,) = model.problem.propagators
        assert propagator[1] == ALG_VALUE_PRECEDE
        assert propagator[2] == [1, 2]  # params [s, t]

    def test_value_precede_solves(self) -> None:
        # value 1 must precede value 2: of the 9 (a, b) pairs over {0,1,2}, 4 are excluded -> 5 remain,
        # and a (the first position) can never be 2
        out = solve_fzn(
            "var 0..2: a :: output_var;\nvar 0..2: b :: output_var;\n"
            "array [1..2] of var int: x = [a, b];\n"
            "constraint fzn_value_precede_int(1, 2, x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 5
        assert "a = 2;" not in out

    def test_value_precede_chain_maps_to_pairwise(self) -> None:
        # a 3-value chain posts one value_precede propagator per consecutive pair
        model = build_model(
            parse("array [1..4] of var 0..3: x;\nconstraint fzn_value_precede_chain_int([1, 2, 3], x);\nsolve satisfy;")
        )
        propagators = model.problem.propagators
        assert [p[1] for p in propagators] == [ALG_VALUE_PRECEDE, ALG_VALUE_PRECEDE]
        assert [p[2] for p in propagators] == [[1, 2], [2, 3]]  # (1 before 2), (2 before 3)

    def test_value_precede_chain_solves(self) -> None:
        # chain [0, 1, 2] over 3 vars: enumerate and compare with the brute-force chain semantics
        from itertools import product

        def valid(xs):  # type: ignore[no-untyped-def]
            return all(xs[:i].count(prev) > 0 for i, v in enumerate(xs) for prev in [v - 1] if 1 <= v <= 2)

        out = solve_fzn(
            "var 0..2: a :: output_var;\nvar 0..2: b :: output_var;\nvar 0..2: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_value_precede_chain_int([0, 1, 2], x);\nsolve satisfy;",
            all_solutions=True,
        )
        expected = sum(1 for xs in product(range(3), repeat=3) if valid(xs))
        assert out.count("----------") == expected

    def test_lex_less_maps_to_lexleq_with_sentinels(self) -> None:
        # a, b are vars 0,1,2,3; lex_less reduces to lexleq over (a ++ [1]) and (b ++ [0]).
        # var_index_of(1)/(0) create the sentinel constants (indices 4, 5 here).
        model = build_model(
            parse(
                "var 0..1: a0;\nvar 0..1: a1;\nvar 0..1: b0;\nvar 0..1: b1;\n"
                "array [1..2] of var int: a = [a0, a1];\narray [1..2] of var int: b = [b0, b1];\n"
                "constraint fzn_lex_less_int(a, b);\nsolve satisfy;"
            )
        )
        (propagator,) = model.problem.propagators
        assert propagator[1] == ALG_LEXLEQ
        one, zero = model.var_index_of(1), model.var_index_of(0)
        assert list(propagator[0]) == [0, 1, one, 2, 3, zero]  # (a ++ [1]) ++ (b ++ [0])

    def test_lex_less_solves(self) -> None:
        from itertools import product

        out = solve_fzn(
            "var 0..1: a0 :: output_var;\nvar 0..1: a1 :: output_var;\n"
            "var 0..1: b0 :: output_var;\nvar 0..1: b1 :: output_var;\n"
            "array [1..2] of var int: a = [a0, a1];\narray [1..2] of var int: b = [b0, b1];\n"
            "constraint fzn_lex_less_int(a, b);\nsolve satisfy;",
            all_solutions=True,
        )
        # count strict-lex pairs over {0,1}^2: (a0,a1) <_lex (b0,b1)
        expected = sum(1 for a in product(range(2), repeat=2) for b in product(range(2), repeat=2) if a < b)
        assert out.count("----------") == expected

    def test_lex_greater_solves_via_swap(self) -> None:
        # MiniZinc lowers lex_greater(a, b) to fzn_lex_less_int(b, a); check the swapped form solves
        from itertools import product

        out = solve_fzn(
            "var 0..1: a0 :: output_var;\nvar 0..1: a1 :: output_var;\n"
            "var 0..1: b0 :: output_var;\nvar 0..1: b1 :: output_var;\n"
            "array [1..2] of var int: a = [a0, a1];\narray [1..2] of var int: b = [b0, b1];\n"
            "constraint fzn_lex_less_int(b, a);\nsolve satisfy;",  # a >_lex b
            all_solutions=True,
        )
        expected = sum(1 for a in product(range(2), repeat=2) for b in product(range(2), repeat=2) if a > b)
        assert out.count("----------") == expected

    def test_increasing_maps_to_increasing(self) -> None:
        model = build_model(parse("array [1..3] of var 0..2: x;\nconstraint fzn_increasing_int(x);\nsolve satisfy;"))
        assert [prop[1] for prop in model.problem.propagators] == [ALG_INCREASING]

    def test_increasing_solve(self) -> None:
        # non-decreasing triples over {0, 1, 2}: there are C(3 + 3 - 1, 3) = 10 of them
        out = solve_fzn(
            "array [1..3] of var 0..2: x :: output_array([1..3]);\nconstraint fzn_increasing_int(x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 10
        assert "x = array1d(1..3, [0, 1, 1]);" in out
        assert "x = array1d(1..3, [2, 1, 0]);" not in out  # not non-decreasing

    def test_strictly_increasing_maps_to_strictly_increasing(self) -> None:
        model = build_model(
            parse("array [1..3] of var 0..4: x;\nconstraint fzn_strictly_increasing_int(x);\nsolve satisfy;")
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_STRICTLY_INCREASING]

    def test_strictly_increasing_solve(self) -> None:
        # strictly increasing triples over {0..4}: there are C(5, 3) = 10 of them
        out = solve_fzn(
            "array [1..3] of var 0..4: x :: output_array([1..3]);\n"
            "constraint fzn_strictly_increasing_int(x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 10
        assert "x = array1d(1..3, [0, 1, 2]);" in out
        assert "x = array1d(1..3, [0, 0, 1]);" not in out  # not strictly increasing

    def test_decreasing_reuses_increasing_on_reversed_list(self) -> None:
        # a, b, c are variables 0, 1, 2; decreasing reuses the increasing propagator over the reversed list
        model = build_model(
            parse("var 0..3: a;\nvar 0..3: b;\nvar 0..3: c;\nconstraint fzn_decreasing_int([a, b, c]);\nsolve satisfy;")
        )
        (propagator,) = model.problem.propagators
        assert propagator[1] == ALG_INCREASING
        assert list(propagator[0]) == [2, 1, 0]  # increasing over [c, b, a] == a >= b >= c

    def test_decreasing_solve(self) -> None:
        # non-increasing triples over {0, 1, 2}: there are C(3 + 3 - 1, 3) = 10 of them
        out = solve_fzn(
            "var 0..2: a :: output_var;\nvar 0..2: b :: output_var;\nvar 0..2: c :: output_var;\n"
            "constraint fzn_decreasing_int([a, b, c]);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 10
        assert "a = 2;\nb = 1;\nc = 0;" in out
        assert "a = 0;\nb = 1;\nc = 2;" not in out  # increasing not allowed

    def test_strictly_decreasing_reuses_strictly_increasing_on_reversed_list(self) -> None:
        model = build_model(
            parse(
                "var 0..4: a;\nvar 0..4: b;\nvar 0..4: c;\n"
                "constraint fzn_strictly_decreasing_int([a, b, c]);\nsolve satisfy;"
            )
        )
        (propagator,) = model.problem.propagators
        assert propagator[1] == ALG_STRICTLY_INCREASING
        assert list(propagator[0]) == [2, 1, 0]

    def test_strictly_decreasing_solve(self) -> None:
        # strictly decreasing triples over {0..4}: there are C(5, 3) = 10 of them
        out = solve_fzn(
            "var 0..4: a :: output_var;\nvar 0..4: b :: output_var;\nvar 0..4: c :: output_var;\n"
            "constraint fzn_strictly_decreasing_int([a, b, c]);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 10
        assert "a = 2;\nb = 1;\nc = 0;" in out

    def test_circuit(self) -> None:
        # 3 nodes, 1-based successors: the only Hamiltonian circuits are the two 3-cycles
        out = solve_fzn(
            "var 1..3: a :: output_var;\nvar 1..3: b :: output_var;\nvar 1..3: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_circuit(x);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # (a,b,c) = (2,3,1) or (3,1,2); the subtour solutions like (1,3,2) are excluded
        assert out.count("----------") == 2
        assert "a = 2;\nb = 3;\nc = 1;" in out and "a = 3;\nb = 1;\nc = 2;" in out

    def test_circuit_zero_based(self) -> None:
        # 3 nodes with 0-based successors (values 0..2): the array values share the model's 0-based node
        # numbering, so _zero_based must NOT shift them. The only Hamiltonian circuits are the two 3-cycles.
        out = solve_fzn(
            "var 0..2: a :: output_var;\nvar 0..2: b :: output_var;\nvar 0..2: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_circuit(x);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        # 0->1->2->0 = (1,2,0) and 0->2->1->0 = (2,0,1); subtour solutions are excluded
        assert out.count("----------") == 2
        assert "a = 1;\nb = 2;\nc = 0;" in out and "a = 2;\nb = 0;\nc = 1;" in out

    def test_circuit_zero_based_uses_variables_directly(self) -> None:
        # values are already 0-based, so no ADD_C_EQ shift is added: ALLDIFFERENT + NO_SUB_CYCLE on x itself
        model = build_model(parse("array [1..3] of var 0..2: x;\nconstraint fzn_circuit(x);\nsolve satisfy;"))
        assert [prop[1] for prop in model.problem.propagators] == [ALG_ALLDIFFERENT, ALG_NO_SUB_CYCLE]

    def test_seq_search_annotation(self) -> None:
        # seq_search nests int_search/bool_search calls inside an array: the parser must accept call terms
        # and the runner must flatten the nested searches into a single decision order.
        out = solve_fzn(
            "var 0..2: x :: output_var;\nvar bool: b :: output_var;\n"
            "constraint int_le(x, 1);\n"
            "solve :: seq_search(["
            "int_search([x], first_fail, indomain_min, complete), "
            "bool_search([b], input_order, indomain_max, complete)]) satisfy;"
        )
        assert "x = 0;" in out

    def test_circuit_self_loop_unsatisfiable(self) -> None:
        # forcing node 1 to point to itself cannot be part of a single circuit over 3 nodes
        out = solve_fzn(
            "var 1..3: a :: output_var;\nvar 1..3: b :: output_var;\nvar 1..3: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_circuit(x);\nconstraint int_eq(a, 1);\n"
            "solve satisfy;"
        )
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_subcircuit_maps_to_alldifferent_and_subcircuit(self) -> None:
        model = build_model(parse("array [1..3] of var 1..3: x;\nconstraint fzn_subcircuit(x);\nsolve satisfy;"))
        # _zero_based adds an ADD_C_EQ per variable, then ALLDIFFERENT + SUBCIRCUIT on the shifted copies
        algorithms = [prop[1] for prop in model.problem.propagators]
        assert algorithms == [ALG_ADD_C_EQ, ALG_ADD_C_EQ, ALG_ADD_C_EQ, ALG_ALLDIFFERENT, ALG_SUBCIRCUIT]

    def test_subcircuit_solves_without_duplicates(self) -> None:
        # over 3 nodes the sub-circuits are: all self-loops, the three 2-cycles, and the two 3-cycles = 6,
        # and crucially each is enumerated exactly once (no auxiliary-variable duplicates)
        out = solve_fzn(
            "var 1..3: a :: output_var;\nvar 1..3: b :: output_var;\nvar 1..3: c :: output_var;\n"
            "array [1..3] of var int: x = [a, b, c];\n"
            "constraint fzn_subcircuit(x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 6
        assert "a = 1;\nb = 2;\nc = 3;" in out  # the empty sub-circuit (all self-loops)
        assert "a = 2;\nb = 1;\nc = 3;" in out  # the 2-cycle (1 2) with 3 excluded

    def test_subcircuit_two_cycles_unsatisfiable(self) -> None:
        # 4 nodes pinned to two disjoint 2-cycles (1 2) and (3 4) is not a single sub-circuit
        out = solve_fzn(
            "array [1..4] of var int: x = [a, b, c, d];\n"
            "var 1..4: a;\nvar 1..4: b;\nvar 1..4: c;\nvar 1..4: d;\n"
            "constraint fzn_subcircuit(x);\n"
            "constraint int_eq(a, 2);\nconstraint int_eq(b, 1);\n"
            "constraint int_eq(c, 4);\nconstraint int_eq(d, 3);\n"
            "solve satisfy;"
        )
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_inverse(self) -> None:
        # 1-based inverse permutations: y is the inverse of x; fixing x pins y
        out = solve_fzn(
            "array [1..3] of var 1..3: x;\narray [1..3] of var 1..3: y :: output_array([1..3]);\n"
            "constraint fzn_inverse(x, y);\n"
            "constraint int_eq(x[1], 2);\nconstraint int_eq(x[2], 3);\nconstraint int_eq(x[3], 1);\n"
            "solve satisfy;"
        )
        # x = [2,3,1] -> inverse y = [3,1,2]
        assert "y = array1d(1..3, [3, 1, 2]);" in out

    def test_inverse_infers_both_ways(self) -> None:
        out = solve_fzn(
            "array [1..3] of var 1..3: x :: output_array([1..3]);\narray [1..3] of var 1..3: y;\n"
            "constraint fzn_inverse(x, y);\n"
            "constraint int_eq(y[1], 2);\nconstraint int_eq(y[2], 3);\nconstraint int_eq(y[3], 1);\n"
            "solve satisfy;"
        )
        # y = [2,3,1] -> x = inverse = [3,1,2]
        assert "x = array1d(1..3, [3, 1, 2]);" in out

    def test_minimize(self) -> None:
        out = solve_fzn(
            "var 0..10: x :: output_var;\nvar 0..10: y :: output_var;\n"
            "constraint int_plus(x, y, 10);\nconstraint int_le(y, x);\n"
            "solve minimize x;"
        )
        assert "x = 5;" in out
        assert out.strip().endswith("==========")

    def test_output_mode_dzn_matches_item(self) -> None:
        fzn = "var 1..3: x :: output_var;\nvar 1..3: y :: output_var;\nconstraint int_lt(x, y);\nsolve satisfy;"
        assert solve_fzn(fzn, output_mode="dzn") == solve_fzn(fzn, output_mode="item")

    def test_output_mode_json(self) -> None:
        out = solve_fzn(
            "var 2..2: x :: output_var;\nvar bool: b :: output_var;\n"
            "array [1..2] of var int: a :: output_array([1..2]) = [x, x];\n"
            "constraint bool_eq(b, true);\nsolve satisfy;",
            output_mode="json",
        )
        # json mode parses to a proper object with bools as true/false and arrays as json arrays
        body = out[: out.index("----------")].strip()
        assert json.loads(body) == {"x": 2, "b": True, "a": [2, 2]}

    def test_output_objective_dzn(self) -> None:
        # maximize x + y under x + y <= 7 -> objective 7
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..9: y :: output_var;\nvar 0..18: z;\n"
            "constraint int_plus(x, y, z);\nconstraint int_le(z, 7);\n"
            "solve maximize z;",
            output_objective=True,
        )
        assert "_objective = 7;" in out

    def test_output_objective_omitted_by_default(self) -> None:
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..18: z;\n"
            "constraint int_le(x, z);\nconstraint int_le(z, 4);\nsolve maximize z;"
        )
        assert "_objective" not in out

    def test_output_objective_json(self) -> None:
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..18: z;\n"
            "constraint int_eq(x, z);\nconstraint int_le(z, 4);\nsolve maximize z;",
            output_mode="json",
            output_objective=True,
        )
        # optimization streams improving solutions; the optimum is the last one before the search-complete marker
        json_blocks = [block.strip() for block in out.split("----------") if block.strip().startswith("{")]
        assert json.loads(json_blocks[-1]) == {"x": 4, "_objective": 4}

    def test_optimization_streams_improving_solutions(self) -> None:
        # input_order/indomain_min finds z=0 first, then branch-and-bound improves it up to the optimum 4;
        # every improving solution is streamed, in non-decreasing objective order, before the search-complete marker
        out = solve_fzn(
            "var 0..9: x :: output_var;\nvar 0..18: z;\n"
            "constraint int_eq(x, z);\nconstraint int_le(z, 4);\nsolve maximize z;",
            output_objective=True,
        )
        objectives = [int(line.split("=")[1].strip(" ;")) for line in out.splitlines() if "_objective" in line]
        assert len(objectives) > 1  # streamed, not just the optimum
        assert objectives == sorted(objectives)  # each solution improves on the previous
        assert objectives[-1] == 4  # the last streamed solution is the optimum
        assert out.rstrip().endswith("==========")  # optimality proven after the last solution

    def test_cli_parses_output_options(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        args = build_arg_parser().parse_args(["model.fzn", "--output-mode", "json", "--output-objective"])
        assert args.output_mode == "json"
        assert args.output_objective is True

    def test_cli_output_mode_defaults_to_item(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        args = build_arg_parser().parse_args(["model.fzn"])
        assert args.output_mode == "item"
        assert args.output_objective is False

    def test_search_heuristics_maps_int_search(self) -> None:
        # x is declared first (index 0), y second (index 1); int_search lists y before x
        model = build_model(
            parse(
                "var 0..3: x;\nvar 0..3: y;\nsolve :: int_search([y, x], first_fail, indomain_max, complete) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result
        assert len(result) == 1  # y and x cover every variable, so there is no trailing catch-all search
        assert result[0].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[0].decision_variables == [1, 0]  # search variables in annotation order (y, x)
        covered = [v for search in result for v in (search.decision_variables or [])]
        assert sorted(covered) == list(range(model.problem.domain_nb))  # every variable is branched

    def test_search_heuristics_none_without_annotation(self) -> None:
        model = build_model(parse("var 0..3: x;\nsolve satisfy;"))
        assert search_heuristics(model) is None

    def test_search_heuristics_unknown_selectors_fall_back(self) -> None:
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], dom_w_deg, indomain_median, complete) satisfy;")
        )
        result = search_heuristics(model)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED  # 'dom_w_deg' has no NuCS equivalent
        assert result[0].dom_heuristic == DOM_HEURISTIC_MID_VALUE

    def test_search_heuristics_maps_smallest_and_largest(self) -> None:
        smallest = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], smallest, indomain_min, complete) satisfy;")
        )
        result = search_heuristics(smallest)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE
        largest = build_model(parse("var 0..3: x;\nsolve :: int_search([x], largest, indomain_min, complete) satisfy;"))
        result = search_heuristics(largest)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE

    def test_search_annotation_value_heuristic_changes_first_solution(self) -> None:
        out = solve_fzn(
            "var 0..2: x :: output_var;\nsolve :: int_search([x], input_order, indomain_max, complete) satisfy;"
        )
        assert "x = 2;" in out  # indomain_max takes the largest value first
        assert "x = 0;" in solve_fzn("var 0..2: x :: output_var;\nsolve satisfy;")  # default is min

    def test_search_heuristics_maps_reverse_split(self) -> None:
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], input_order, indomain_reverse_split, complete) satisfy;")
        )
        result = search_heuristics(model)
        assert result
        assert result[0].dom_heuristic == DOM_HEURISTIC_SPLIT_HIGH

    def test_search_heuristics_seq_search_keeps_groups_and_appends_catch_all(self) -> None:
        # seq_search becomes one Search per nested search (each with its own selectors), plus a catch-all
        model = build_model(
            parse(
                "var 0..3: a;\nvar 0..3: b;\nvar 0..3: c;\n"
                "solve :: seq_search(["
                "int_search([b], input_order, indomain_max, complete),"
                "int_search([a], first_fail, indomain_min, complete)"
                "]) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result is not None
        assert len(result) == 3  # the two nested searches and a catch-all for c
        assert result[0].decision_variables == [1]  # b
        assert result[0].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[1].decision_variables == [0]  # a
        assert result[1].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[1].dom_heuristic == DOM_HEURISTIC_MIN_VALUE
        assert result[2].decision_variables == [2]  # c, the remaining variable, with the defaults
        assert result[2].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
        assert result[2].dom_heuristic == DOM_HEURISTIC_MIN_VALUE

    def test_search_heuristics_flattens_nested_seq_search(self) -> None:
        # MiniZinc nests a seq_search when decomposing e.g. set_search; the nested searches must be flattened
        # in order instead of dropped (which would lose their selectors)
        model = build_model(
            parse(
                "var 0..3: a;\nvar 0..3: b;\n"
                "solve :: seq_search(["
                "int_search([a], input_order, indomain_max, complete),"
                "seq_search([int_search([b], first_fail, indomain_min, complete)])"
                "]) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result is not None
        assert len(result) == 2  # a and b cover every variable, so no catch-all
        assert result[0].decision_variables == [0]  # a
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[1].decision_variables == [1]  # b, lifted out of the nested seq_search
        assert result[1].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[1].dom_heuristic == DOM_HEURISTIC_MIN_VALUE

    def test_search_annotation_reverse_split_takes_upper_half_first(self) -> None:
        out = solve_fzn(
            "var 0..3: x :: output_var;\n"
            "solve :: int_search([x], input_order, indomain_reverse_split, complete) satisfy;"
        )
        assert "x = 3;" in out  # reverse split explores the upper half first

    def test_all_solutions_terminator(self) -> None:
        out = solve_fzn(
            "var 1..3: x :: output_var;\nvar 1..3: y :: output_var;\nconstraint int_lt(x, y);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert out.strip().endswith("==========")

    def test_unsatisfiable(self) -> None:
        out = solve_fzn("var 1..2: x :: output_var;\nconstraint int_eq(x, 5);\nsolve satisfy;")
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_unsupported_builtin(self) -> None:
        with pytest.raises(FznUnsupportedError):
            build_model(parse("var 0..1: x;\nvar 0..1: y;\nconstraint int_pow(x, y, x);\nsolve satisfy;"))

    def test_var_times_var(self) -> None:
        out = solve_fzn(
            "var 2..2: x :: output_var;\nvar 3..3: y :: output_var;\nvar 0..81: z :: output_var;\n"
            "constraint int_times(x, y, z);\nsolve satisfy;"
        )
        assert "z = 6;" in out  # 2 * 3 = 6

    def test_int_mod(self) -> None:
        out = solve_fzn(
            "var 17..17: x :: output_var;\nvar 5..5: y :: output_var;\nvar -20..20: z :: output_var;\n"
            "constraint int_mod(x, y, z);\nsolve satisfy;"
        )
        assert "z = 2;" in out  # 17 mod 5 = 2

    def test_int_mod_negative_dividend(self) -> None:
        # truncated division: the remainder takes the sign of the dividend, so -17 mod 5 = -2
        out = solve_fzn(
            "var -17..-17: x :: output_var;\nvar 5..5: y :: output_var;\nvar -20..20: z :: output_var;\n"
            "constraint int_mod(x, y, z);\nsolve satisfy;"
        )
        assert "z = -2;" in out

    def test_int_mod_constant_divisor_uses_mod_c_eq(self) -> None:
        # a literal divisor routes to the bound-consistent constant-modulus propagator (no divisor variable)
        model = build_model(parse("var 0..9: x;\nvar -9..9: z;\nconstraint int_mod(x, 5, z);\nsolve satisfy;"))
        algorithms = [prop[1] for prop in model.problem.propagators]
        assert algorithms == [ALG_MOD_C_EQ]
        assert model.problem.propagators[0][2] == [5]  # params = [modulus]

    def test_int_mod_variable_divisor_uses_mod_eq(self) -> None:
        model = build_model(
            parse("var 0..9: x;\nvar 1..9: y;\nvar -9..9: z;\nconstraint int_mod(x, y, z);\nsolve satisfy;")
        )
        assert [prop[1] for prop in model.problem.propagators] == [ALG_MOD_EQ]

    def test_int_mod_constant_divisor_prunes_dividend(self) -> None:
        # the constant-modulus propagator also prunes x to the residue class fixed by z (x mod 7 = 3)
        out = solve_fzn(
            "var 0..100: x :: output_var;\nvar 3..3: z :: output_var;\n"
            "constraint int_mod(x, 7, z);\n"
            "solve :: int_search([x], input_order, indomain_min, complete) satisfy;"
        )
        assert "x = 3;" in out  # smallest non-negative x with x mod 7 = 3
