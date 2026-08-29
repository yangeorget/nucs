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
Run all NuCS examples and produce an aggregated performance report.

Usage:
    NUMBA_CACHE_DIR=.numba/cache python scripts/benchmark.py
    NUMBA_CACHE_DIR=.numba/cache python scripts/benchmark.py --json logs/run.json
    NUMBA_CACHE_DIR=.numba/cache python scripts/benchmark.py --only queens_10 --only golomb_10

The first run may include JIT compilation time; subsequent runs use the cache.

Each model runs in its own child process. That is what makes the peak-RSS column meaningful: the OS
high-water mark is monotone within a process, so measuring it per model requires a fresh one. It also
keeps a model that dies from taking the rest of the run with it. --in-process skips the isolation (and
the RSS column) when only the timings are wanted.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from nucs.constants import (
    MAX,
    MIN,
    OPTIM_PRUNE,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_ELAPSED_TIME,
)
from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem
from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.examples.langford.langford_problem import LangfordProblem
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.examples.magic_square.magic_square_problem import MagicSquareProblem
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.examples.social_golfers.social_golfers_problem import SocialGolfersProblem
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.examples.tsp.tsp_var_heuristic import tsp_var_heuristic
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MIN_COST,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    register_var_heuristic,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm

# name, statistics, domain_nb, choice-point bytes
BenchmarkResult = tuple[str, dict[str, int], int, int]

STATS_LBL_DOMAIN_NB = "DOMAIN_NB"
STATS_LBL_PEAK_RSS_KB = "PEAK_RSS_KB"
STATS_LBL_CHOICE_POINT_BYTES = "CHOICE_POINT_BYTES"

# the solver attributes holding backtrackable search state, whichever of them the branch under measurement
# happens to define -- this is the allocation the migration is meant to shrink
CHOICE_POINT_ARRAYS = (
    "domains_stk",
    "domain_update_stk",
    "unbound_variable_nb_stk",
    "entailment_trail",
    "entailed_propagator_depths",
    "state",
    "trail_log",
    "trail_indices",
    "choice_point_stk",
    "entailed",
)


def _choice_point_bytes(solver: BacktrackSolver) -> int:
    """
    Returns the bytes the solver preallocates for backtrackable search state.

    RSS alone understates it: np.empty reserves address space that only becomes resident as the search
    deepens, so a model that never goes deep never pays for the ceiling it nevertheless imposes. This is
    the number the memory argument is actually about.

    :param solver: the solver
    :type solver: BacktrackSolver

    :return: the number of bytes
    :rtype: int
    """
    return sum(
        getattr(solver, name).nbytes
        for name in CHOICE_POINT_ARRAYS
        if isinstance(getattr(solver, name, None), np.ndarray)
    )


def _result(name: str, solver: BacktrackSolver) -> BenchmarkResult:
    return name, solver.get_statistics_as_dictionary(), solver.problem.domain_nb, _choice_point_bytes(solver)


def solve_all(name: str, solver: BacktrackSolver) -> BenchmarkResult:
    solver.solve_all()
    return _result(name, solver)


def first_solution(name: str, solver: BacktrackSolver) -> BenchmarkResult:
    next(solver.solve(), None)
    return _result(name, solver)


def minimize(name: str, solver: BacktrackSolver, variable: int) -> BenchmarkResult:
    solver.find_best(variable, MIN, OPTIM_PRUNE)
    return _result(name, solver)


def maximize(name: str, solver: BacktrackSolver, variable: int) -> BenchmarkResult:
    solver.find_best(variable, MAX)
    return _result(name, solver)


def _benchmarks() -> dict[str, Callable[[], BenchmarkResult]]:
    def queens_10() -> BenchmarkResult:
        return solve_all("queens(10)", BacktrackSolver(QueensProblem(10), log_level="WARNING"))

    def queens_11() -> BenchmarkResult:
        return solve_all("queens(11)", BacktrackSolver(QueensProblem(11), log_level="WARNING"))

    def queens_12() -> BenchmarkResult:
        return solve_all("queens(12)", BacktrackSolver(QueensProblem(12), log_level="WARNING"))

    def queens_13() -> BenchmarkResult:
        return solve_all("queens(13)", BacktrackSolver(QueensProblem(13), log_level="WARNING"))

    def golomb_9() -> BenchmarkResult:
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(9)
        return minimize(
            "golomb(9)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def golomb_10() -> BenchmarkResult:
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(10)
        return minimize(
            "golomb(10)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def golomb_11() -> BenchmarkResult:
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(11)
        return minimize(
            "golomb(11)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def magic_sequence_100() -> BenchmarkResult:
        return solve_all(
            "magic_sequence(100)",
            BacktrackSolver(MagicSequenceProblem(100), decision_variables=range(99, -1, -1), log_level="WARNING"),
        )

    def magic_sequence_200() -> BenchmarkResult:
        return solve_all(
            "magic_sequence(200)",
            BacktrackSolver(MagicSequenceProblem(200), decision_variables=range(199, -1, -1), log_level="WARNING"),
        )

    def magic_square_3() -> BenchmarkResult:
        return solve_all(
            "magic_square(3)",
            BacktrackSolver(
                MagicSquareProblem(3),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_MAX_VALUE,
                log_level="WARNING",
            ),
        )

    def magic_square_4() -> BenchmarkResult:
        return solve_all(
            "magic_square(4)",
            BacktrackSolver(
                MagicSquareProblem(4),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_MAX_VALUE,
                log_level="WARNING",
            ),
        )

    def all_interval_10() -> BenchmarkResult:
        return solve_all("all_interval(10)", BacktrackSolver(AllIntervalSeriesProblem(10, True), log_level="WARNING"))

    def all_interval_11() -> BenchmarkResult:
        return solve_all("all_interval(11)", BacktrackSolver(AllIntervalSeriesProblem(11, True), log_level="WARNING"))

    def all_interval_12() -> BenchmarkResult:
        return solve_all("all_interval(12)", BacktrackSolver(AllIntervalSeriesProblem(12, True), log_level="WARNING"))

    def langford_2_9() -> BenchmarkResult:
        return solve_all("langford(2,9)", BacktrackSolver(LangfordProblem(2, 9), log_level="WARNING"))

    def langford_3_9() -> BenchmarkResult:
        return solve_all("langford(3,9)", BacktrackSolver(LangfordProblem(3, 9), log_level="WARNING"))

    def bibd_7() -> BenchmarkResult:
        return solve_all("bibd(7,7,3,3,1)", BacktrackSolver(BIBDProblem(7, 7, 3, 3, 1), log_level="WARNING"))

    def bibd_8() -> BenchmarkResult:
        return solve_all("bibd(8,14,7,4,3)", BacktrackSolver(BIBDProblem(8, 14, 7, 4, 3), log_level="WARNING"))

    def golfers_3_2_5() -> BenchmarkResult:
        return solve_all("golfers(3,2,5)", BacktrackSolver(SocialGolfersProblem(3, 2, 5, True), log_level="WARNING"))

    def golfers_3_3_4() -> BenchmarkResult:
        return solve_all("golfers(3,3,4)", BacktrackSolver(SocialGolfersProblem(3, 3, 4, True), log_level="WARNING"))

    def quasigroup_3_8() -> BenchmarkResult:
        return solve_all(
            "quasigroup(3,8)",
            BacktrackSolver(
                QuasigroupProblem(3, 8, True),
                decision_variables=range(64),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
                log_level="WARNING",
            ),
        )

    def quasigroup_5_10() -> BenchmarkResult:
        return solve_all(
            "quasigroup(5,10)",
            BacktrackSolver(
                QuasigroupProblem(5, 10, True),
                decision_variables=range(10 * 10),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
                log_level="WARNING",
            ),
        )

    def quasigroup_5_11() -> BenchmarkResult:
        return solve_all(
            "quasigroup(5,11)",
            BacktrackSolver(
                QuasigroupProblem(5, 11, True),
                decision_variables=range(11 * 11),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
                log_level="WARNING",
            ),
        )

    def quasigroup_5_12() -> BenchmarkResult:
        return solve_all(
            "quasigroup(5,12)",
            BacktrackSolver(
                QuasigroupProblem(5, 12, True),
                decision_variables=range(12 * 12),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
                log_level="WARNING",
            ),
        )

    def tsp_gr17() -> BenchmarkResult:
        with open("datasets/examples/tsp/gr17.json") as json_file:
            costs = json.load(json_file)["costs"]
        n = len(costs)
        problem = TSPProblem(costs)
        costs = costs + costs
        tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
        return minimize(
            "tsp(gr17)",
            BacktrackSolver(
                problem,
                decision_variables=range(2 * n),
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
                log_level="WARNING",
            ),
            problem.total_cost,
        )

    def tsp_gr21() -> BenchmarkResult:
        with open("datasets/examples/tsp/gr21.json") as json_file:
            costs = json.load(json_file)["costs"]
        n = len(costs)
        problem = TSPProblem(costs)
        costs = costs + costs
        tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
        return minimize(
            "tsp(gr21)",
            BacktrackSolver(
                problem,
                decision_variables=range(2 * n),
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
                log_level="WARNING",
            ),
            problem.total_cost,
        )

    def bibd_10() -> BenchmarkResult:
        # the shape the rest of the set lacks: domain_nb in the hundreds *and* a node count in the tens of
        # thousands, which is the only regime where a per-node whole-domains copy is on the critical path
        return solve_all("bibd(10,15,6,4,2)", BacktrackSolver(BIBDProblem(10, 15, 6, 4, 2), log_level="WARNING"))

    return {
        function.__name__: function
        for function in [
            all_interval_10,
            all_interval_11,
            all_interval_12,
            bibd_7,
            bibd_8,
            bibd_10,
            golfers_3_2_5,
            golfers_3_3_4,
            golomb_9,
            golomb_10,
            golomb_11,
            langford_2_9,
            langford_3_9,
            magic_sequence_100,
            magic_sequence_200,
            magic_square_3,
            magic_square_4,
            quasigroup_3_8,
            quasigroup_5_10,
            quasigroup_5_11,
            quasigroup_5_12,
            queens_10,
            queens_11,
            queens_12,
            queens_13,
            tsp_gr17,
            tsp_gr21,
        ]
    }


def _run_one(key: str) -> dict[str, Any]:
    """
    Runs one benchmark in the current process and returns its statistics.

    :param key: the name of the benchmark function
    :type key: str

    :return: the statistics, augmented with the domain nb and the peak RSS of the process
    :rtype: Dict[str, Any]
    """
    name, statistics, domain_nb, choice_point_bytes = _benchmarks()[key]()
    result: dict[str, Any] = {"name": name, **statistics}
    result[STATS_LBL_DOMAIN_NB] = domain_nb
    result[STATS_LBL_CHOICE_POINT_BYTES] = choice_point_bytes
    # ru_maxrss is bytes on macOS and kibibytes on Linux
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result[STATS_LBL_PEAK_RSS_KB] = max_rss // 1024 if sys.platform == "darwin" else max_rss
    return result


def _run_isolated(key: str) -> dict[str, Any]:
    """
    Runs one benchmark in a child process, so its peak RSS is its own rather than the whole run's.

    :param key: the name of the benchmark function
    :type key: str

    :return: the statistics reported by the child
    :rtype: Dict[str, Any]
    """
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", key],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


def _ratio(numerator: int, denominator: int, pct: bool = False) -> str:
    if denominator == 0:
        return "-"
    ratio = numerator / denominator
    return f"{ratio * 100:.1f}%" if pct else f"{ratio:.2f}"


def _int(value: int) -> str:
    return f"{value:,}"


def _report(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="\nNuCS Benchmark Report", show_lines=False, header_style="bold cyan")
    table.add_column("Example", style="bold", no_wrap=True)
    table.add_column("Vars", justify="right")
    table.add_column("Solutions", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("CP alloc (MB)", justify="right")
    table.add_column("Peak RSS (MB)", justify="right")
    table.add_column("Backtracks", justify="right")
    table.add_column("BC calls", justify="right")
    table.add_column("Propagations", justify="right")
    table.add_column("Entailments", justify="right")
    table.add_column("bt/ms", justify="right", style="yellow")
    table.add_column("prop/BC", justify="right", style="yellow")
    table.add_column("useless%", justify="right", style="yellow")
    table.add_column("incons%", justify="right", style="yellow")
    table.add_column("entail%", justify="right", style="yellow")
    for statistics in results:
        backtracks = statistics[STATS_LBL_SOLVER_BACKTRACK_NB]
        milliseconds = statistics[STATS_LBL_SOLVER_ELAPSED_TIME]
        bc_calls = statistics[STATS_LBL_ALG_BC_NB]
        propagations = statistics[STATS_LBL_PROPAGATOR_FILTER_NB]
        no_change = statistics[STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB]
        inconsistencies = statistics[STATS_LBL_PROPAGATOR_INCONSISTENCY_NB]
        entailments = statistics[STATS_LBL_PROPAGATOR_ENTAILMENT_NB]
        peak_rss = statistics.get(STATS_LBL_PEAK_RSS_KB)
        table.add_row(
            statistics["name"],
            _int(statistics[STATS_LBL_DOMAIN_NB]),
            _int(statistics[STATS_LBL_SOLUTION_NB]),
            _int(milliseconds),
            f"{statistics[STATS_LBL_CHOICE_POINT_BYTES] / (1024 * 1024):.1f}",
            "-" if peak_rss is None else f"{peak_rss / 1024:.0f}",
            _int(backtracks),
            _int(bc_calls),
            _int(propagations),
            _int(entailments),
            _ratio(backtracks, milliseconds),
            _ratio(propagations, bc_calls),
            _ratio(no_change, propagations, pct=True),
            _ratio(inconsistencies, propagations, pct=True),
            _ratio(entailments, propagations, pct=True),
        )
    console.print(table)
    console.print(
        "\n[dim]"
        "Vars     = domain_nb, the number of variables the solver allocates state for\n"
        "CP alloc = the bytes preallocated for backtrackable state, the ceiling this work removes\n"
        "Peak RSS = the child process high-water mark; it lags CP alloc, which np.empty only reserves\n"
        "bt/ms    = backtracks per millisecond\n"
        "prop/BC  = propagator filter calls per bound consistency computation\n"
        "useless% = filter calls that changed nothing\n"
        "incons%  = filter calls that detected an inconsistency\n"
        "entail%  = filter calls that detected entailment"
        "[/dim]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--only", action="append", default=[], help="run only these benchmarks, repeatable")
    parser.add_argument("--in-process", action="store_true", help="skip the per-model subprocess, and the RSS column")
    parser.add_argument("--json", help="also write the raw results to this file")
    args = parser.parse_args()
    if args.child:  # a child process: run the one benchmark and report it as JSON on stdout
        print(json.dumps(_run_one(args.child)))
        return
    console = Console(width=200)
    keys = args.only or list(_benchmarks())
    results: list[dict[str, Any]] = []
    console.print("\n[bold]Running NuCS benchmarks…[/bold]\n")
    for key in keys:
        try:
            statistics = _run_one(key) if args.in_process else _run_isolated(key)
            results.append(statistics)
            name = statistics["name"]
            milliseconds = statistics[STATS_LBL_SOLVER_ELAPSED_TIME]
            solutions = statistics[STATS_LBL_SOLUTION_NB]
            console.print(f"  [green]✓[/green] {name:<30}  {solutions:>6} solution(s)  {milliseconds:>6} ms")
        except Exception as exception:  # noqa: BLE001 - one broken model must not end the run
            console.print(f"  [red]✗[/red] {key:<30}  {exception}")
    _report(console, results)
    if args.json:
        with open(args.json, "w") as json_file:
            json.dump(results, json_file, indent=2)
        console.print(f"[dim]raw results written to {args.json}[/dim]")


if __name__ == "__main__":
    main()
