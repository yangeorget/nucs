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
End-to-end tests driving the real ``minizinc`` binary with the NuCS solver.

These are skipped unless ``minizinc`` is on PATH (or pointed to by the ``MINIZINC`` env var); they are the
first-milestone gate and are meant to be run on a machine with MiniZinc installed.
"""

import os
import shutil
import subprocess

import pytest

from nucs.fzn.builtins import BUILTINS

SHARE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nucs", "fzn", "share")
GLOBALS_LIB = os.path.join(SHARE_DIR, "minizinc", "nucs")

MINIZINC = os.environ.get("MINIZINC") or shutil.which("minizinc")

# global name as used in a model -> (model snippet, predicate that must survive in the FlatZinc,
#                                    redefinition file that keeps it native)
KEPT_GLOBALS = {
    "all_different": (
        "array[1..4] of var 1..4: q; constraint all_different(q);",
        "fzn_all_different_int",
        "fzn_all_different_int.mzn",
    ),
    "circuit": (
        "array[1..4] of var 1..4: s; constraint circuit(s);",
        "nucs_circuit",
        "fzn_circuit.mzn",
    ),
    "count_eq": (
        "array[1..4] of var 1..4: x; constraint count_eq(x, 2, 2);",
        "fzn_count_eq",
        "fzn_count_eq.mzn",
    ),
    "count_geq": (
        "array[1..4] of var 1..4: x; constraint count_geq(x, 2, 2);",
        "fzn_count_geq",
        "fzn_count_geq.mzn",
    ),
    "count_leq": (
        "array[1..4] of var 1..4: x; constraint count_leq(x, 2, 2);",
        "fzn_count_leq",
        "fzn_count_leq.mzn",
    ),
    "diffn": (
        "array[1..3] of var 0..9: x; array[1..3] of var 0..9: y; constraint diffn(x, y, [2, 3, 1], [2, 1, 2]);",
        "nucs_diffn",
        "fzn_diffn.mzn",
    ),
    "disjunctive": (
        "array[1..3] of var 0..9: s; constraint disjunctive(s, [2, 3, 1]);",
        "nucs_disjunctive",
        "fzn_disjunctive_strict.mzn",
    ),
    # global_cardinality_low_up is deprecated; the modern global_cardinality(x, cover, lb, ub) lowers to
    # the same fzn_global_cardinality_low_up predicate that NuCS keeps native.
    "global_cardinality": (
        "array[1..4] of var 1..4: x; constraint global_cardinality(x,[1,2,3,4],[0,0,0,0],[2,2,2,2]);",
        "fzn_global_cardinality_low_up",
        "fzn_global_cardinality_low_up.mzn",
    ),
    "increasing": (
        "array[1..4] of var 0..9: x; constraint increasing(x);",
        "fzn_increasing_int",
        "fzn_increasing_int.mzn",
    ),
    "inverse": (
        "array[1..3] of var 1..3: f; array[1..3] of var 1..3: g; constraint inverse(f,g);",
        "nucs_inverse",
        "fzn_inverse.mzn",
    ),
    "lex_less": (
        "array[1..3] of var 0..2: a; array[1..3] of var 0..2: b; constraint lex_less(a,b);",
        "fzn_lex_less_int",
        "fzn_lex_less_int.mzn",
    ),
    "lex_lesseq": (
        "array[1..3] of var 0..2: a; array[1..3] of var 0..2: b; constraint lex_lesseq(a,b);",
        "fzn_lex_lesseq_int",
        "fzn_lex_lesseq_int.mzn",
    ),
    "nvalue": (
        "var 0..4: n; array[1..4] of var 0..3: x; constraint nvalue(n, x);",
        "fzn_nvalue",
        "fzn_nvalue.mzn",
    ),
    "strictly_increasing": (
        "array[1..4] of var 0..9: x; constraint strictly_increasing(x);",
        "fzn_strictly_increasing_int",
        "fzn_strictly_increasing_int.mzn",
    ),
    "subcircuit": (
        "array[1..4] of var 1..4: x; constraint subcircuit(x);",
        "nucs_subcircuit",
        "fzn_subcircuit.mzn",
    ),
    "table": (
        "array[1..2] of var 0..2: t; constraint table(t, [| 0,1 | 1,2 |]);",
        "nucs_table_int",
        "fzn_table_int.mzn",
    ),
    "value_precede": (
        "array[1..4] of var 0..3: x; constraint value_precede(1, 2, x);",
        "fzn_value_precede_int",
        "fzn_value_precede_int.mzn",
    ),
    "value_precede_chain": (
        "array[1..4] of var 0..3: x; constraint value_precede_chain([1, 2, 3], x);",
        "fzn_value_precede_chain_int",
        "fzn_value_precede_chain_int.mzn",
    ),
}


def _compile_to_fzn(model: str, tmp_path) -> str:  # type: ignore[no-untyped-def]
    """Compiles a MiniZinc model to FlatZinc with the NuCS solver, returning the FlatZinc text."""
    assert MINIZINC is not None
    model_path = tmp_path / "model.mzn"
    model_path.write_text(f'include "globals.mzn";\n{model}\nsolve satisfy;\n')
    env = dict(os.environ, MZN_SOLVER_PATH=SHARE_DIR)
    result = subprocess.run(
        [MINIZINC, "-c", "--solver", "nucs", str(model_path), "--output-fzn-to-stdout"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _solve(model: str, tmp_path) -> str:  # type: ignore[no-untyped-def]
    """Solves a MiniZinc model with the NuCS solver, returning the solution stream text."""
    assert MINIZINC is not None
    model_path = tmp_path / "model.mzn"
    model_path.write_text(f'include "globals.mzn";\n{model}\n')
    env = dict(os.environ, MZN_SOLVER_PATH=SHARE_DIR)
    result = subprocess.run(
        [MINIZINC, "--solver", "nucs", str(model_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("name", sorted(KEPT_GLOBALS))
def test_global_stays_native(name, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The NuCS globals library keeps each kept global native instead of decomposing it."""
    model, predicate, _ = KEPT_GLOBALS[name]
    fzn = _compile_to_fzn(model, tmp_path)
    assert f"constraint {predicate}(" in fzn, f"{name} was decomposed instead of kept native:\n{fzn}"


def test_circuit_non_one_based_index_with_wide_domain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """circuit over an index set 2..5 whose declared domain (-100..100) is wider than the node set: the
    fzn_circuit library rebases successor values to the index set, so the solution is a circuit on 2..5
    (not the garbage negative values produced when the offset was guessed from the variable domain)."""
    out = _solve(
        "array[2..5] of var -100..100: c;\nconstraint circuit(c);\n"
        "solve :: int_search(c, input_order, indomain_min, complete) satisfy;",
        tmp_path,
    )
    # the lexicographically first circuit on nodes 2..5 explored with indomain_min
    assert "c = [2: 3, 3: 4, 4: 5, 5: 2];" in out


def test_subcircuit_non_one_based_index_with_wide_domain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """subcircuit over an index set 2..5 with a wide declared domain: like circuit, successor values are
    rebased to the node set, so self-loops and the sub-circuit stay within 2..5."""
    out = _solve(
        "array[2..5] of var -100..100: c;\nconstraint subcircuit(c);\n"
        "solve :: int_search(c, input_order, indomain_min, complete) satisfy;",
        tmp_path,
    )
    # indomain_min first grounds every node to a self-loop (the empty sub-circuit)
    assert "c = [2: 2, 3: 3, 4: 4, 5: 5];" in out


def test_inverse_non_one_based_index_with_wide_domain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """inverse over index sets 2..4 whose declared domains (-100..100) are wider than the node sets: the
    fzn_inverse library rebases each array's values to the other's index set, so g is the genuine inverse of
    f on nodes 2..4 (not garbage from an offset guessed off the variable domain)."""
    out = _solve(
        "array[2..4] of var -100..100: f;\narray[2..4] of var -100..100: g;\n"
        "constraint inverse(f, g);\nconstraint f[2] = 3;\nconstraint f[3] = 4;\nconstraint f[4] = 2;\n"
        "solve satisfy;",
        tmp_path,
    )
    assert "f = [2: 3, 3: 4, 4: 2];" in out
    assert "g = [2: 4, 3: 2, 4: 3];" in out  # the inverse permutation, on nodes 2..4


def test_every_redefinition_file_is_covered() -> None:
    """Fails when a kept-global redefinition exists in the library but no test exercises it.

    Only the per-global ``fzn_*.mzn`` native-predicate files are kept-globals; ``redefinitions.mzn`` is a
    catch-all auto-included file (it supplies overloads, e.g. the non-optional strictly_decreasing), not a
    kept global, so it is excluded.
    """
    on_disk = {f for f in os.listdir(GLOBALS_LIB) if f.startswith("fzn_") and f.endswith(".mzn")}
    covered = {entry[2] for entry in KEPT_GLOBALS.values()}
    assert on_disk == covered, f"uncovered: {sorted(on_disk - covered)}; stale: {sorted(covered - on_disk)}"


@pytest.mark.parametrize("name", sorted(KEPT_GLOBALS))
def test_native_predicate_is_dispatched(name) -> None:  # type: ignore[no-untyped-def]
    """Each native predicate the globals library emits has a handler in the BUILTINS dispatch table."""
    _, predicate, _ = KEPT_GLOBALS[name]
    assert predicate in BUILTINS


def test_strictly_decreasing_reaches_native_strictly_increasing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The redefinitions.mzn non-optional overload routes strictly_decreasing to NuCS's native
    strictly_increasing propagator (over the reversed array) instead of an int_lin_le decomposition chain."""
    fzn = _compile_to_fzn("array[1..4] of var 0..9: x; constraint strictly_decreasing(x);", tmp_path)
    assert "constraint fzn_strictly_increasing_int(" in fzn
    assert "int_lin_le" not in fzn


def test_set_variable_is_decomposed_into_bools(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The nosets.mzn include rewrites a var set into an array of var bool, so no var set reaches NuCS
    (which has no set variables); the model flattens instead of failing with 'domain set is not supported'."""
    fzn = _compile_to_fzn("var set of 1..4: s; constraint 2 in s; constraint card(s) = 2;", tmp_path)
    assert "var set" not in fzn
    assert "var bool" in fzn


def test_max_reaches_native_array_int_maximum(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The redefinitions-2.0.mzn body-less declaration keeps the n-ary max native (NuCS's MAX_EQ propagator)
    instead of decomposing it into a chain of binary int_max constraints with one introduced variable each."""
    fzn = _compile_to_fzn("array[1..4] of var 0..9: x; var int: m = max(x);", tmp_path)
    assert "constraint array_int_maximum(" in fzn
    assert "constraint int_max(" not in fzn


def test_min_reaches_native_array_int_minimum(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The redefinitions-2.0.mzn body-less declaration keeps the n-ary min native (NuCS's MIN_EQ propagator)
    instead of decomposing it into a chain of binary int_min constraints with one introduced variable each."""
    fzn = _compile_to_fzn("array[1..4] of var 0..9: x; var int: m = min(x);", tmp_path)
    assert "constraint array_int_minimum(" in fzn
    assert "constraint int_min(" not in fzn
