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

The first run may include JIT compilation time; subsequent runs use the cache.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from nucs.constants import (
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_ELAPSED_TIME, OPTIM_PRUNE, )
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
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_SMALLEST_DOMAIN, register_var_heuristic, DOM_HEURISTIC_MIN_COST,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm

BenchmarkResult = Tuple[str, Dict[str, int], Optional[Any]]


def solve_all(name: str, solver: BacktrackSolver) -> BenchmarkResult:
    solver.solve_all()
    return name, solver.get_statistics_as_dictionary(), None


def first_solution(name: str, solver: BacktrackSolver) -> BenchmarkResult:
    solution = next(solver.solve(), None)
    return name, solver.get_statistics_as_dictionary(), solution


def minimize(name: str, solver: BacktrackSolver, variable: int) -> BenchmarkResult:
    solution = solver.minimize(variable, mode=OPTIM_PRUNE)
    return name, solver.get_statistics_as_dictionary(), solution


def maximize(name: str, solver: BacktrackSolver, variable: int) -> BenchmarkResult:
    solution = solver.maximize(variable)
    return name, solver.get_statistics_as_dictionary(), solution


def _benchmarks() -> List[Callable[[], BenchmarkResult]]:
    def queens_10():
        return solve_all("queens(10)", BacktrackSolver(QueensProblem(10), log_level="WARNING"))

    def queens_11():
        return solve_all("queens(11)", BacktrackSolver(QueensProblem(11), log_level="WARNING"))

    def queens_12():
        return solve_all("queens(12)", BacktrackSolver(QueensProblem(12), log_level="WARNING"))

    def queens_13():
        return solve_all("queens(13)", BacktrackSolver(QueensProblem(13), log_level="WARNING"))

    def golomb_9():
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(9)
        return minimize(
            "golomb(9)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def golomb_10():
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(10)
        return minimize(
            "golomb(10)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def golomb_11():
        alg = register_consistency_algorithm(golomb_consistency_algorithm)
        problem = GolombProblem(11)
        return minimize(
            "golomb(11)",
            BacktrackSolver(problem, consistency_algorithm=alg, log_level="WARNING"),
            problem.length_idx,
        )

    def magic_sequence_100():
        return solve_all(
            "magic_sequence(100)",
            BacktrackSolver(MagicSequenceProblem(100), decision_variables=range(99, -1, -1), log_level="WARNING"),
        )

    def magic_sequence_200():
        return solve_all(
            "magic_sequence(200)",
            BacktrackSolver(MagicSequenceProblem(200), decision_variables=range(199, -1, -1), log_level="WARNING"),
        )

    def magic_square_3():
        return solve_all(
            "magic_square(3)",
            BacktrackSolver(
                MagicSquareProblem(3),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_MAX_VALUE,
                log_level="WARNING",
            ),
        )

    def magic_square_4():
        return solve_all(
            "magic_square(4)",
            BacktrackSolver(
                MagicSquareProblem(4),
                var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic=DOM_HEURISTIC_MAX_VALUE,
                log_level="WARNING",
            ),
        )

    def all_interval_10():
        return solve_all(
            "all_interval(10)", BacktrackSolver(AllIntervalSeriesProblem(10, True), log_level="WARNING")
        )

    def all_interval_11():
        return solve_all(
            "all_interval(11)", BacktrackSolver(AllIntervalSeriesProblem(11, True), log_level="WARNING")
        )

    def all_interval_12():
        return solve_all(
            "all_interval(12)", BacktrackSolver(AllIntervalSeriesProblem(12, True), log_level="WARNING")
        )

    def langford_2_9():
        return solve_all("langford(2,9)", BacktrackSolver(LangfordProblem(2, 9), log_level="WARNING"))

    def langford_3_9():
        return solve_all("langford(3,9)", BacktrackSolver(LangfordProblem(3, 9), log_level="WARNING"))

    def bibd_7():
        return solve_all("bibd(7,7,3,3,1)", BacktrackSolver(BIBDProblem(7, 7, 3, 3, 1), log_level="WARNING"))

    def bibd_8():
        return solve_all("bibd(8,14,7,4,3)", BacktrackSolver(BIBDProblem(8, 14, 7, 4, 3), log_level="WARNING"))

    def golfers_3_2_5():
        return solve_all(
            "golfers(3,2,5)", BacktrackSolver(SocialGolfersProblem(3, 2, 5, True), log_level="WARNING")
        )

    def golfers_3_3_4():
        return solve_all(
            "golfers(3,3,4)", BacktrackSolver(SocialGolfersProblem(3, 3, 4, True), log_level="WARNING")
        )

    def quasigroup_5_10():
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

    def quasigroup_5_11():
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

    def quasigroup_5_12():
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

    def quasigroup_3_8():
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

    def tsp_gr17():
        with open("datasets/tsp/gr17.json", "r") as json_file:
            costs = json.load(json_file)["costs"]
            n = len(costs)
            problem = TSPProblem(costs)
            costs = costs + costs
            tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
            return minimize("tsp(gr17)", BacktrackSolver(
                problem,
                decision_variables=range(0, 2 * n),
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
            ), problem.total_cost)

    def tsp_gr21():
        with open("datasets/tsp/gr21.json", "r") as json_file:
            costs = json.load(json_file)["costs"]
            n = len(costs)
            problem = TSPProblem(costs)
            costs = costs + costs
            tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
            return minimize("tsp(gr21)", BacktrackSolver(
                problem,
                decision_variables=range(0, 2 * n),
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
            ), problem.total_cost)

    return [
        all_interval_10,
        all_interval_11,
        all_interval_12,
        bibd_7,
        bibd_8,
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
        tsp_gr21
    ]


def _ratio(numerator: int, denominator: int, pct: bool = False) -> str:
    if denominator == 0:
        return "-"
    r = numerator / denominator
    return f"{r * 100:.1f}%" if pct else f"{r:.2f}"


def _int(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    console = Console(width=180)
    results: List[BenchmarkResult] = []

    console.print("\n[bold]Running NuCS benchmarks…[/bold]\n")
    for bench in _benchmarks():
        try:
            result = bench()
            results.append(result)
            name, stats, _ = result
            ms = stats[STATS_LBL_SOLVER_ELAPSED_TIME]
            sols = stats[STATS_LBL_SOLUTION_NB]
            console.print(f"  [green]✓[/green] {name:<30}  {sols:>6} solution(s)  {ms:>6} ms")
        except Exception as exc:
            console.print(f"  [red]✗[/red] {bench.__name__:<30}  {exc}")

    table = Table(title="\nNuCS Benchmark Report", show_lines=False, header_style="bold cyan")
    table.add_column("Example", style="bold", no_wrap=True)
    table.add_column("Solutions", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Backtracks", justify="right")
    table.add_column("BC calls", justify="right")
    table.add_column("Propagations", justify="right")
    table.add_column("Entailments", justify="right")
    table.add_column("bt/ms", justify="right", style="yellow")
    table.add_column("prop/BC", justify="right", style="yellow")
    table.add_column("useless%", justify="right", style="yellow")
    table.add_column("incons%", justify="right", style="yellow")
    table.add_column("entail%", justify="right", style="yellow")

    for name, stats, _ in results:
        bt = stats[STATS_LBL_SOLVER_BACKTRACK_NB]
        ms = stats[STATS_LBL_SOLVER_ELAPSED_TIME]
        bc = stats[STATS_LBL_ALG_BC_NB]
        props = stats[STATS_LBL_PROPAGATOR_FILTER_NB]
        no_change = stats[STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB]
        incons = stats[STATS_LBL_PROPAGATOR_INCONSISTENCY_NB]
        entail = stats[STATS_LBL_PROPAGATOR_ENTAILMENT_NB]
        sols = stats[STATS_LBL_SOLUTION_NB]

        table.add_row(
            name,
            _int(sols),
            _int(ms),
            _int(bt),
            _int(bc),
            _int(props),
            _int(entail),
            _ratio(bt, ms),
            _ratio(props, bc),
            _ratio(no_change, props, pct=True),
            _ratio(incons, props, pct=True),
            _ratio(entail, props, pct=True),
        )

    console.print(table)
    console.print(
        "\n[dim]"
        "bt/ms   = backtracks per millisecond\n"
        "prop/BC = propagator filter calls per bound consistency computation\n"
        "useless% = filter calls that changed nothing\n"
        "incons% = filter calls that detected an inconsistency\n"
        "entail% = filter calls that detected entailment"
        "[/dim]\n"
    )


if __name__ == "__main__":
    main()
