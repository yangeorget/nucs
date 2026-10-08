"""
Solves one problem of chart 3 with the NuCS that is first on sys.path, and prints one line: RESULT {json}.

collect_data.py runs it in the worktree of each release tag. It uses only the API that every measured tag has:
the example problems, BacktrackSolver(problem), solve() and get_statistics_as_dictionary(). Each problem uses the
default search, because the way to give a search changed in 18.0.0. golomb_N and tsp_N are minimizations, in the PRUNE
mode: up to v14.1.0 through solver.minimize, from v15.0.0 through solver.find_best. tsp_N uses the first N cities of
gr17 (data/tsp_gr17.json), so that every tag reads the same matrix.

The first solve compiles or loads the jitted code and is not timed. Then the problem is solved REPEATS[problem] times,
and the time of one solve is the total divided by that number: a short problem is repeated so that the timer and the
noise stay small in front of the measured time.

The logs are turned off: the solver logs during the search, and the cost of a log depends on where the output goes (a
pipe, a file, a terminal), not on the solver. scripts/benchmark.py does the same with log_level="WARNING".
"""

import json
import logging
import sys
import time
from pathlib import Path

import nucs
from nucs.solvers.backtrack_solver import BacktrackSolver


REPEATS = {"queens_12": 1, "all_interval_13": 1, "bibd_10": 20}
OPTIMIZATION = ("golomb_", "tsp_")  # problem name prefixes: the problem is a minimization


def build(name: str):  # noqa: ANN201 -- the problem class depends on the tag
    if name == "queens_12":
        from nucs.examples.queens.queens_problem import QueensProblem

        return QueensProblem(12)
    if name == "all_interval_13":
        from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem

        return AllIntervalSeriesProblem(13, True)
    if name == "bibd_10":
        from nucs.examples.bibd.bibd_problem import BIBDProblem

        return BIBDProblem(10, 15, 6, 4, 2)
    if name.startswith("golomb_"):
        from nucs.examples.golomb.golomb_problem import GolombProblem

        problem = GolombProblem(int(name.split("_")[1]))
        return problem, problem.length_idx
    if name.startswith("tsp_"):
        from nucs.examples.tsp.tsp_problem import TSPProblem

        n = int(name.split("_")[1])
        costs = json.loads((Path(__file__).parent / "data" / "tsp_gr17.json").read_text())["costs"]
        problem = TSPProblem([row[:n] for row in costs[:n]])
        return problem, problem.total_cost
    raise ValueError(name)


def minimize(solver, variable: int):  # noqa: ANN201 -- the solution type depends on the tag
    """The best solution in the PRUNE mode; the bound 0 is the minimum (DOMAIN_MIN, MIN before 16.0.0)."""
    if hasattr(solver, "find_best"):
        return solver.find_best(variable, 0, "PRUNE")
    return solver.minimize(variable, "PRUNE")


def run(name: str) -> tuple[int, float, dict]:
    """For a minimization, the first value returned is the optimum, so that every tag can be checked against it."""
    if name.startswith(OPTIMIZATION):
        problem, variable = build(name)
        solver = BacktrackSolver(problem)
        start = time.perf_counter()
        solution = minimize(solver, variable)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return int(solution[variable]), elapsed_ms, solver.get_statistics_as_dictionary()
    solver = BacktrackSolver(build(name))
    start = time.perf_counter()
    solution_nb = sum(1 for _ in solver.solve())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return solution_nb, elapsed_ms, solver.get_statistics_as_dictionary()


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)  # every tag logs through the standard logging module
    problem = sys.argv[1]
    run(problem)  # the warm-up
    total_ms = 0.0
    for _ in range(REPEATS.get(problem, 1)):
        solution_nb, elapsed_ms, statistics = run(problem)
        total_ms += elapsed_ms
    elapsed_ms = total_ms / REPEATS.get(problem, 1)
    print("RESULT " + json.dumps({
        "problem": problem,
        "time_ms": round(elapsed_ms, 2),
        "repeats": REPEATS.get(problem, 1),
        "solutions": solution_nb,
        "statistics": {k: int(v) for k, v in statistics.items()},
        "nucs_file": nucs.__file__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))
