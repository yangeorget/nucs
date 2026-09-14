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
Run the FlatZinc benchmark models and report, per model, which propagators they actually exercise.

Usage:
    NUMBA_CACHE_DIR=.numba/cache python scripts/fzn_benchmark.py
    NUMBA_CACHE_DIR=.numba/cache python scripts/fzn_benchmark.py --only gcc_roster --timeout 30
    NUMBA_CACHE_DIR=.numba/cache python scripts/fzn_benchmark.py --coverage --json logs/fzn.json

The models in ``datasets/fzn`` exist to cover what the Python examples in ``nucs/examples`` do not.
Most of NuCS's globals are reachable only through FlatZinc -- ``value_precede_chain``, ``bin_packing_load``,
a ``count`` over a long array, ``regular``, ``nvalue`` -- and a propagator no model posts cannot be
measured, which repeatedly left changes to those propagators undecidable rather than decided. Hence
``--coverage``, which answers the question those investigations kept needing: for each propagator, is
there a model that calls it, and how hot is it.

Each model is compiled with ``minizinc --solver nucs`` (so it goes through NuCS's own globals library)
into a ``.fzn`` cached beside it, and is then solved in a child process, which is what makes the timeout
enforceable and keeps one model's failure off the rest of the run.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "datasets" / "fzn"
MINIZINC = os.environ.get("MINIZINC") or shutil.which("minizinc")


def annotation_of(mzn: Path, key: str) -> str | None:
    """
    Returns a ``% nucs-<key>: value`` annotation from a model, or None when it declares none.

    :param mzn: the model to read
    :type mzn: Path
    :param key: the annotation name, without the ``nucs-`` prefix
    :type key: str

    :return: the annotation's value
    :rtype: Optional[str]
    """
    prefix = f"% nucs-{key}:"
    for line in mzn.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def target_of(mzn: Path) -> str | None:
    """
    Returns the propagator a model exists to exercise, declared in it as ``% nucs-target: NAME``.

    A model is worth keeping only if it actually calls the propagator it was written for, and the hottest
    three are often the reifications and linear terms FlatZinc lowers everything else into. Declaring the
    target is what lets the report say whether the model still earns its place.

    :param mzn: the model to read
    :type mzn: Path

    :return: the algorithm name, or None when the model declares none
    :rtype: Optional[str]
    """
    return annotation_of(mzn, "target")


def compile_model(mzn: Path, rebuild: bool = False) -> Path:
    """
    Compiles a model to FlatZinc through NuCS's globals library, caching the result beside the source.

    :param mzn: the model to compile
    :type mzn: Path
    :param rebuild: whether to compile even when the cached FlatZinc is up to date
    :type rebuild: bool

    :return: the path of the compiled FlatZinc
    :rtype: Path
    """
    if MINIZINC is None:
        raise RuntimeError("minizinc is not on PATH; set MINIZINC or install it")
    fzn = mzn.with_suffix(".fzn")
    dzn = mzn.with_suffix(".dzn")
    fresh = not rebuild and fzn.exists() and fzn.stat().st_mtime >= mzn.stat().st_mtime
    if fresh and (not dzn.exists() or fzn.stat().st_mtime >= dzn.stat().st_mtime):
        return fzn
    command = [MINIZINC, "-c", "--solver", "nucs", str(mzn)]
    if dzn.exists():
        command += ["-d", str(dzn)]
    command += ["--output-fzn-to-file", str(fzn)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return fzn


def solve(fzn: Path, all_solutions: bool = False) -> dict[str, Any]:
    """
    Solves one compiled model and returns its statistics, including the per-propagator call counts.

    :param fzn: the compiled FlatZinc to solve
    :type fzn: Path
    :param all_solutions: whether to enumerate every solution rather than stop at the first. A satisfiable
        model is often solved greedily, which exercises its propagators barely at all; enumerating makes
        the search depth a property of the instance size, and so tunable
    :type all_solutions: bool

    :return: the model's statistics, its propagator counts by algorithm name, and its arities
    :rtype: Dict[str, Any]
    """
    from nucs.fzn.model import build_model
    from nucs.fzn.parser import parse
    from nucs.fzn.runner import search_heuristics
    from nucs.problems.problem import OFFSETS_VARIABLE
    from nucs.propagators.propagators import get_algorithm_names
    from nucs.solvers.backtrack_solver import BacktrackSolver
    from nucs.statistics import STATS_ALG_IDX_FILTER_NB, STATS_ALG_IDX_FILTER_NO_CHANGE_NB, STATS_ALG_WIDTH, STATS_MAX

    model = build_model(parse(fzn.read_text()))
    problem = model.problem
    searches = search_heuristics(model)
    solver = (
        BacktrackSolver(problem, log_level="ERROR")
        if searches is None
        else BacktrackSolver(problem, searches=searches, log_level="ERROR")
    )
    objective = None if model.solve.objective is None else model.var_index_of(model.solve.objective)
    started = time.perf_counter()
    if model.solve.kind == "satisfy":
        if all_solutions:
            found = sum(1 for _ in solver.solve()) > 0
        else:
            found = next(solver.solve(), None) is not None
    else:
        from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
        from nucs.solvers.solver import OPTIM_PRUNE

        bound = DOMAIN_MIN if model.solve.kind == "minimize" else DOMAIN_MAX
        assert objective is not None
        found = False
        for _ in solver.optimize(objective, bound, OPTIM_PRUNE):
            found = True
    elapsed = time.perf_counter() - started
    names = get_algorithm_names()
    # arity is a property of the posted propagator, not of the algorithm, so the widest one posted is what
    # says whether this model exercises an algorithm at the size where its cost lives
    arity: dict[str, int] = {}
    for propagator in range(problem.propagator_nb):
        name = names[problem.algorithms[propagator]]
        width = int(problem.offsets[propagator + 1, OFFSETS_VARIABLE] - problem.offsets[propagator, OFFSETS_VARIABLE])
        arity[name] = max(arity.get(name, 0), width)
    calls = {}
    for algorithm, name in enumerate(names):
        base = STATS_MAX + STATS_ALG_WIDTH * algorithm
        count = int(solver.statistics[base + STATS_ALG_IDX_FILTER_NB])
        if count:
            calls[name] = (count, int(solver.statistics[base + STATS_ALG_IDX_FILTER_NO_CHANGE_NB]), arity.get(name, 0))
    return {
        "solved": found,
        "elapsed_ms": round(elapsed * 1000, 1),
        "variables": int(problem.domain_nb),
        "propagators": int(problem.propagator_nb),
        "statistics": solver.get_statistics_as_dictionary(),
        "calls": calls,
    }


def _run_isolated(mzn: Path, timeout: float) -> dict[str, Any]:
    """
    Runs one model in a child process, so that a timeout can be enforced and a crash contained.

    :param mzn: the model to run
    :type mzn: Path
    :param timeout: how many seconds to allow
    :type timeout: float

    :return: the child's result, or a record saying it timed out or failed
    :rtype: Dict[str, Any]
    """
    try:
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", str(mzn)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR", ".numba/cache")},
        )
    except subprocess.TimeoutExpired:
        return {"status": f"timeout >{timeout:g}s"}
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        return {"status": "FAILED: " + (tail[-1][:80] if tail else "no output")}
    return {"status": "ok", **json.loads(completed.stdout.strip().splitlines()[-1])}


def _report(console: Console, results: list[tuple[str, dict[str, Any]]]) -> None:
    table = Table(title="FlatZinc benchmark models", title_justify="left")
    columns = ("model", "vars", "time", "nodes", "target propagator", "calls", "nc", "arity", "hottest")
    for column in columns:
        table.add_column(
            column, justify="right" if column in ("vars", "time", "nodes", "calls", "nc", "arity") else "left"
        )
    for name, result in results:
        if result["status"] != "ok":
            table.add_row(name, "-", result["status"], *["-"] * 6)
            continue
        target = result.get("target")
        entry = result["calls"].get(target) if target else None
        hot = sorted(result["calls"].items(), key=lambda item: -item[1][0])[:1]
        table.add_row(
            name,
            f"{result['variables']:,}",
            f"{result['elapsed_ms']:,.0f} ms",
            f"{result['statistics'].get('ALG_BC_NB', 0):,}",
            target or "-",
            f"{entry[0]:,}" if entry else "[red]0[/red]",
            f"{100 * entry[1] / entry[0]:.0f}%" if entry else "-",
            str(entry[2]) if entry else "-",
            "  ".join(f"{n}={c:,}" for n, (c, _, _) in hot),
        )
    console.print(table)


def _coverage(console: Console, results: list[tuple[str, dict[str, Any]]]) -> None:
    from nucs.propagators.propagators import get_algorithm_names

    totals: dict[str, tuple[int, int]] = {}
    for _, result in results:
        if result["status"] != "ok":
            continue
        for name, (count, _, wide) in result["calls"].items():
            previous = totals.get(name, (0, 0))
            totals[name] = (previous[0] + count, max(previous[1], wide))
    table = Table(title="Propagator coverage", title_justify="left")
    table.add_column("propagator")
    table.add_column("calls", justify="right")
    table.add_column("widest posted", justify="right")
    for name in sorted(get_algorithm_names()):
        if name == "DUMMY":
            continue
        calls, wide = totals.get(name, (0, 0))
        style = None if calls else "dim red"
        table.add_row(name, f"{calls:,}" if calls else "never called", str(wide) if calls else "-", style=style)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--only", action="append", default=[], help="run only these models, repeatable")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds allowed per model")
    parser.add_argument("--rebuild", action="store_true", help="recompile the FlatZinc even when it is up to date")
    parser.add_argument("--coverage", action="store_true", help="also report which propagators went uncalled")
    parser.add_argument("--json", help="also write the raw results to this file")
    args = parser.parse_args()
    if args.child:
        child = Path(args.child)
        every = annotation_of(child, "solve") == "all"
        print(json.dumps({**solve(compile_model(child), every), "target": target_of(child)}))
        return
    console = Console(width=200)
    models = sorted(MODEL_DIR.glob("*.mzn"))
    if args.only:
        models = [m for m in models if m.stem in args.only]
    results = []
    for mzn in models:
        compile_model(mzn, args.rebuild)
        results.append((mzn.stem, _run_isolated(mzn, args.timeout)))
    _report(console, results)
    if args.coverage:
        _coverage(console, results)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(results), indent=1))


if __name__ == "__main__":
    main()
