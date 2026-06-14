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
        "fzn_circuit",
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
        "fzn_inverse",
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
        "fzn_subcircuit",
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


@pytest.mark.parametrize("name", sorted(KEPT_GLOBALS))
def test_global_stays_native(name, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The NuCS globals library keeps each kept global native instead of decomposing it."""
    model, predicate, _ = KEPT_GLOBALS[name]
    fzn = _compile_to_fzn(model, tmp_path)
    assert f"constraint {predicate}(" in fzn, f"{name} was decomposed instead of kept native:\n{fzn}"


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
