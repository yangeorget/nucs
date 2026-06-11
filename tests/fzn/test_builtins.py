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
from typing import Optional

import pytest

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run
from nucs.propagators.propagators import ALG_LINEAR_EQ_C, ALG_ALLDIFFERENT, ALG_LEQ_C


def solve_fzn(fzn: str, all_solutions: bool = False, num_solutions: Optional[int] = None) -> str:
    """Builds, solves and returns the FlatZinc solution stream as text."""
    out = io.StringIO()
    run(
        build_model(parse(fzn)),
        out,
        io.StringIO(),
        all_solutions=all_solutions,
        num_solutions=num_solutions,
    )
    return out.getvalue()


class TestBuiltins:
    def test_mapping_propagators(self) -> None:
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\n"
                "constraint int_lin_eq([1, 1], [x, y], 5);\n"
                "constraint int_le(x, y);\n"
                "constraint all_different_int([x, y]);\n"
                "solve satisfy;"
            )
        )
        algorithms = [prop[1] for prop in model.problem.propagators]
        assert algorithms == [ALG_LINEAR_EQ_C, ALG_LEQ_C, ALG_ALLDIFFERENT]
        # int_lin_eq([1,1], [x,y], 5) -> linear_eq_c with params coeffs + [c]
        assert model.problem.propagators[0][2] == [1, 1, 5]

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

    def test_minimize(self) -> None:
        out = solve_fzn(
            "var 0..10: x :: output_var;\nvar 0..10: y :: output_var;\n"
            "constraint int_plus(x, y, 10);\nconstraint int_le(y, x);\n"
            "solve minimize x;"
        )
        assert "x = 5;" in out
        assert out.strip().endswith("==========")

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
