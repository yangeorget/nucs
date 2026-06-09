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

SHARE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nucs", "fzn", "share")

MINIZINC = os.environ.get("MINIZINC") or shutil.which("minizinc")

ALL_DIFFERENT_LINEAR_MODEL = """
include "globals.mzn";
array [1..4] of var 1..4: q;
constraint all_different(q);
constraint sum(q) = 10;
solve satisfy;
output [show(q)];
"""


pytestmark = pytest.mark.skipif(MINIZINC is None, reason="minizinc binary not available")


def _run_minizinc(model: str, tmp_path) -> str:  # type: ignore[no-untyped-def]
    """Compiles and solves a MiniZinc model with the NuCS solver, returning stdout."""
    assert MINIZINC is not None
    model_path = tmp_path / "model.mzn"
    model_path.write_text(model)
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


def test_all_different_linear(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = _run_minizinc(ALL_DIFFERENT_LINEAR_MODEL, tmp_path)
    assert "----------" in out
