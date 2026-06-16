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
Diagnoses why a MiniZinc/FlatZinc problem is slow through the NuCS FlatZinc adapter.

It separates the three cost centers of the MiniZinc -> FlatZinc -> NuCS pipeline:
  1. MZN -> FZN flattening time (the ``minizinc`` binary),
  2. the FlatZinc model MiniZinc emitted (which globals stayed native vs were decomposed into primitives),
  3. the NuCS model built from it (variables, propagator mix) and -- optionally -- the search effort.

The decisive signal is the propagator mix: a model dominated by ``linear_neq_c`` / ``*_reif`` / ``*_element``
where the source model used ``all_different`` / ``cumulative`` / ``table`` means a strong global was
decomposed into weak primitives, which is what makes the search explode.

Usage::

    python scripts/diagnose_fzn.py model.mzn [data.dzn ...]   # flattens then analyzes
    python scripts/diagnose_fzn.py model.fzn                  # analyzes an existing FlatZinc
    python scripts/diagnose_fzn.py model.mzn data.dzn --solve --time-limit 30

Run with the NuCS environment (so ``minizinc --solver nucs`` resolves and ``import nucs`` works).
"""

import argparse
import collections
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import List, Tuple

import nucs
from nucs.constants import (
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
)
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS
from nucs.solvers.backtrack_solver import BacktrackSolver

SHARE_DIR = os.path.join(os.path.dirname(nucs.__file__), "fzn", "share")

# A FlatZinc constraint whose name starts with one of these is a primitive (scalar/array/set/bool builtin),
# not a global. Everything else (all_different_int, fzn_cumulative, table_int, global_cardinality, ...) is a
# global that MiniZinc kept native rather than decomposing.
_PRIMITIVE_PREFIXES = ("int_", "bool_", "float_", "set_", "array_")


def _flatten(mzn: str, data: List[str]) -> Tuple[str, float]:
    """Flattens a MiniZinc model with the NuCS solver, returning the FlatZinc text and the elapsed time."""
    env = dict(os.environ, MZN_SOLVER_PATH=SHARE_DIR)
    cmd = ["minizinc", "-c", "--solver", "nucs", mzn, *data, "--output-fzn-to-stdout"]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        sys.exit(f"minizinc flattening failed:\n{result.stderr}")
    return result.stdout, elapsed


def _constraint_histogram(fzn: str) -> "collections.Counter[str]":
    """Counts FlatZinc constraints by builtin name."""
    counter: collections.Counter[str] = collections.Counter()
    for line in fzn.splitlines():
        line = line.strip()
        if line.startswith("constraint "):
            rest = line[len("constraint ") :].lstrip()
            name = rest.split("(", 1)[0].split(":", 1)[0].strip()
            if name:
                counter[name] += 1
    return counter


def _print_histogram(counter: "collections.Counter[str]", top: int) -> None:
    """Prints a count-sorted histogram, truncated to the top entries."""
    items = counter.most_common(top)
    width = max((len(name) for name, _ in items), default=0)
    for name, count in items:
        print(f"  {count:8d}  {name:<{width}}")
    remaining = len(counter) - len(items)
    if remaining > 0:
        print(f"  {'':8}  ... and {remaining} more kinds")


def _solve_worker(problem, model, queue) -> None:  # type: ignore[no-untyped-def]
    """Solves to the first solution / optimum and reports stats on the queue. Run in a child process so it can
    be hard-killed on timeout (the search runs in compiled code and ignores in-process signals)."""
    solver = BacktrackSolver(problem, log_level="ERROR")
    if model.solve.kind in ("minimize", "maximize"):
        objective_var = model.var_index_of(model.solve.objective)
        solution = (
            solver.minimize(objective_var) if model.solve.kind == "minimize" else solver.maximize(objective_var)
        )
        status = "OPTIMAL" if solution is not None else "UNSATISFIABLE"
        objective = int(solution[objective_var]) if solution is not None else None
    else:
        solution = next(solver.solve(), None)
        status = "SATISFIABLE" if solution is not None else "UNSATISFIABLE"
        objective = None
    queue.put(
        (
            status,
            objective,
            int(solver.statistics[STATS_IDX_SOLVER_BACKTRACK_NB]),
            int(solver.statistics[STATS_IDX_PROPAGATOR_FILTER_NB]),
        )
    )


def _solve_with_timeout(problem, model, seconds: int) -> None:  # type: ignore[no-untyped-def]
    """Runs the solve in a forked child and hard-kills it after the timeout. With ``fork`` the child inherits
    the built model and the warm JIT cache (no recompile); on timeout it is terminated, so no NuCS stats are
    available -- the search was still running in compiled code."""
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_solve_worker, args=(problem, model, queue))
    start = time.perf_counter()
    process.start()
    process.join(seconds)
    elapsed = time.perf_counter() - start
    if process.is_alive():
        process.terminate()
        process.join()
        print(f"  status      : TIMEOUT (killed at {seconds}s)")
        print(f"  solve time  : >{seconds}s (search still running in compiled code -- no stats)")
        return
    status, objective, backtracks, filters = queue.get()
    print(f"  status      : {status}")
    if objective is not None:
        print(f"  objective   : {objective}")
    print(f"  solve time  : {elapsed:.3f}s")
    print(f"  backtracks  : {backtracks}")
    print(f"  filter calls: {filters}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NuCS FlatZinc-adapter performance for a problem.")
    parser.add_argument("model", help="a .mzn model (flattened with minizinc) or an existing .fzn file")
    parser.add_argument("data", nargs="*", help="optional .dzn data files (only with a .mzn model)")
    parser.add_argument("--solve", action="store_true", help="also solve to the first solution / optimum")
    parser.add_argument("--time-limit", type=int, default=30, help="solve timeout in seconds (default 30)")
    parser.add_argument("--top", type=int, default=15, help="how many histogram entries to show (default 15)")
    args = parser.parse_args()

    print(f"== INPUT ==\n  {args.model}" + (f"  +  {', '.join(args.data)}" if args.data else ""))

    if args.model.endswith(".fzn"):
        with open(args.model) as f:
            fzn = f.read()
    else:
        fzn, flatten_time = _flatten(args.model, args.data)
        print(f"\n== FLATTEN (mzn -> fzn) ==\n  flatten time: {flatten_time:.3f}s")

    constraints = _constraint_histogram(fzn)
    globals_kept = collections.Counter(
        {n: c for n, c in constraints.items() if not n.startswith(_PRIMITIVE_PREFIXES)}
    )
    primitives = sum(constraints.values()) - sum(globals_kept.values())
    print(f"\n== FLATZINC EMITTED ({sum(constraints.values())} constraints) ==")
    print(f"  primitive constraints: {primitives}")
    _print_histogram(constraints, args.top)
    print("  globals kept native:", (dict(globals_kept) if globals_kept else "NONE — every global was decomposed"))

    start = time.perf_counter()
    model = build_model(parse(fzn))
    build_time = time.perf_counter() - start
    mix = collections.Counter(
        COMPUTE_DOMAINS_FCTS[prop[1]].__name__.replace("compute_domains_", "") for prop in model.problem.propagators
    )
    print(f"\n== NuCS MODEL ==\n  build time : {build_time:.3f}s")
    print(f"  variables  : {len(model.problem.domains)}")
    print(f"  propagators: {len(model.problem.propagators)}")
    print("  propagator mix (ALG -> count):")
    _print_histogram(mix, args.top)

    if args.solve:
        print(f"\n== SOLVE (first solution / optimum, timeout {args.time_limit}s) ==")
        _solve_with_timeout(model.problem, model, args.time_limit)


if __name__ == "__main__":
    main()
