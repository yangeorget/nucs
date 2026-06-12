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
    "global_cardinality_low_up": (
        "array[1..4] of var 1..4: x; constraint global_cardinality_low_up(x,[1,2,3,4],[0,0,0,0],[2,2,2,2]);",
        "fzn_global_cardinality_low_up",
        "fzn_global_cardinality_low_up.mzn",
    ),
    "inverse": (
        "array[1..3] of var 1..3: f; array[1..3] of var 1..3: g; constraint inverse(f,g);",
        "fzn_inverse",
        "fzn_inverse.mzn",
    ),
    "lex_lesseq": (
        "array[1..3] of var 0..2: a; array[1..3] of var 0..2: b; constraint lex_lesseq(a,b);",
        "fzn_lex_lesseq_int",
        "fzn_lex_lesseq_int.mzn",
    ),
    "table": (
        "array[1..2] of var 0..2: t; constraint table(t, [| 0,1 | 1,2 |]);",
        "nucs_table_int",
        "fzn_table_int.mzn",
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
    """Fails when a kept-global redefinition exists in the library but no test exercises it."""
    on_disk = {f for f in os.listdir(GLOBALS_LIB) if f.endswith(".mzn")}
    covered = {entry[2] for entry in KEPT_GLOBALS.values()}
    assert on_disk == covered, f"uncovered: {sorted(on_disk - covered)}; stale: {sorted(covered - on_disk)}"


@pytest.mark.parametrize("name", sorted(KEPT_GLOBALS))
def test_native_predicate_is_dispatched(name) -> None:  # type: ignore[no-untyped-def]
    """Each native predicate the globals library emits has a handler in the BUILTINS dispatch table."""
    _, predicate, _ = KEPT_GLOBALS[name]
    assert predicate in BUILTINS
