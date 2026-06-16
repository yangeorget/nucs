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
End-to-end cross-checks: for well-defined problems, the result obtained by solving the native NuCS model
must match the result obtained through the MiniZinc -> FlatZinc -> NuCS pipeline.

The native side uses the example ``Problem`` classes directly; the pipeline side compiles a faithful MiniZinc
model to FlatZinc with the real ``minizinc`` binary and solves that FlatZinc with the in-process NuCS runner.
For satisfaction problems we compare the total number of solutions (relabeling-invariant); for optimization
problems we compare the optimum (symmetry-invariant). Expected values are not hardcoded -- each test computes
both sides and asserts they agree.

These tests are skipped unless ``minizinc`` is on PATH (or pointed to by the ``MINIZINC`` env var).
"""

import io
import os
import re
import shutil
import subprocess
import tempfile

import pytest

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.examples.langford.langford_problem import LangfordProblem
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm

SHARE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nucs", "fzn", "share")
MINIZINC = os.environ.get("MINIZINC") or shutil.which("minizinc")

pytestmark = pytest.mark.skipif(MINIZINC is None, reason="minizinc is not installed")


def _compile_to_fzn(model: str) -> str:
    """Compiles a full MiniZinc model (with its own solve item) to FlatZinc with the NuCS solver."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.mzn")
        with open(model_path, "w") as f:
            f.write(model)
        env = dict(os.environ, MZN_SOLVER_PATH=SHARE_DIR)
        assert MINIZINC
        result = subprocess.run(
            [MINIZINC, "-c", "--solver", "nucs", model_path, "--output-fzn-to-stdout"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _pipeline_solution_count(model: str) -> int:
    """Returns the number of solutions found by the MiniZinc -> FlatZinc -> NuCS pipeline."""
    fzn = _compile_to_fzn(model)
    out = io.StringIO()
    run(build_model(parse(fzn)), out, io.StringIO(), all_solutions=True)
    return out.getvalue().count("----------")


def _pipeline_optimum(model: str) -> int:
    """Returns the objective of the optimal solution found by the pipeline."""
    fzn = _compile_to_fzn(model)
    out = io.StringIO()
    run(build_model(parse(fzn)), out, io.StringIO(), output_objective=True)
    matches = re.findall(r"objective = (-?\d+);", out.getvalue())
    assert matches, f"no objective in solution stream:\n{out.getvalue()}"
    return int(matches[-1])


def _native_solution_count(problem) -> int:  # type: ignore[no-untyped-def]
    """Returns the total number of solutions of a native NuCS problem."""
    solver = BacktrackSolver(problem)
    solver.solve_all()
    return int(solver.statistics[STATS_IDX_SOLUTION_NB])


class TestEndToEnd:
    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_queens_count(self, n: int) -> None:
        model = (
            'include "globals.mzn";\n'
            f"int: n = {n};\n"
            "array[1..n] of var 1..n: q;\n"
            "constraint all_different(q);\n"
            "constraint all_different([q[i] + i | i in 1..n]);\n"
            "constraint all_different([q[i] - i | i in 1..n]);\n"
            "solve satisfy;\n"
        )
        assert _pipeline_solution_count(model) == _native_solution_count(QueensProblem(n))

    @pytest.mark.parametrize("mark_nb", [4, 5, 6])
    def test_golomb_optimum(self, mark_nb: int) -> None:
        model = (
            'include "globals.mzn";\n'
            f"int: n = {mark_nb};\n"
            f"int: maxlen = {mark_nb * mark_nb};\n"
            "array[1..n] of var 0..maxlen: mark;\n"
            "constraint mark[1] = 0;\n"
            "constraint forall(i in 1..n-1)(mark[i] < mark[i+1]);\n"
            "constraint all_different([mark[j] - mark[i] | i in 1..n-1, j in i+1..n]);\n"
            "solve minimize mark[n];\n"
        )
        problem = GolombProblem(mark_nb)
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        solution = BacktrackSolver(problem, consistency_algorithm=alg).minimize(problem.length_idx)
        assert solution is not None
        assert _pipeline_optimum(model) == solution[problem.length_idx]

    @pytest.mark.parametrize("n", [5, 6, 7, 8])
    def test_magic_sequence_count(self, n: int) -> None:
        model = (
            'include "globals.mzn";\n'
            f"int: n = {n};\n"
            "array[0..n-1] of var 0..n-1: x;\n"
            "constraint forall(i in 0..n-1)(count(x, i) = x[i]);\n"
            "solve satisfy;\n"
        )
        assert _pipeline_solution_count(model) == _native_solution_count(MagicSequenceProblem(n))

    @pytest.mark.parametrize("k,n", [(2, 3), (2, 4), (2, 7)])
    def test_langford_count(self, k: int, n: int) -> None:
        model = (
            'include "globals.mzn";\n'
            f"int: k = {k};\nint: n = {n};\n"
            "array[1..k, 1..n] of var 1..k*n: pos;\n"
            "constraint all_different([pos[i, j] | i in 1..k, j in 1..n]);\n"
            "constraint forall(j in 1..n, i in 1..k-1)(pos[i + 1, j] = pos[i, j] + j + 1);\n"
            "solve satisfy;\n"
        )
        assert _pipeline_solution_count(model) == _native_solution_count(LangfordProblem(k, n))
