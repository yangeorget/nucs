"""
Solves one problem of chart 3 with the NuCS that is first on sys.path, and prints one line: RESULT {json}.

collect_data.py runs it in the worktree of each release tag. It uses only the API that every measured tag has:
the example problems, BacktrackSolver(problem), solve() and get_statistics_as_dictionary(). Each problem uses the
default search, because the way to give a search changed in 18.0.0.

The first solve compiles or loads the jitted code and is not timed. Then the problem is solved REPEATS[problem] times,
and the time of one solve is the total divided by that number: a short problem is repeated so that the timer and the
noise stay small in front of the measured time.
"""

import json
import sys
import time

import nucs
from nucs.solvers.backtrack_solver import BacktrackSolver


REPEATS = {"queens_12": 1, "all_interval_13": 1, "bibd_10": 20}


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
    raise ValueError(name)


def run(name: str) -> tuple[int, float, dict]:
    solver = BacktrackSolver(build(name))
    start = time.perf_counter()
    solution_nb = sum(1 for _ in solver.solve())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return solution_nb, elapsed_ms, solver.get_statistics_as_dictionary()


if __name__ == "__main__":
    problem = sys.argv[1]
    run(problem)  # the warm-up
    total_ms = 0.0
    for _ in range(REPEATS[problem]):
        solution_nb, elapsed_ms, statistics = run(problem)
        total_ms += elapsed_ms
    elapsed_ms = total_ms / REPEATS[problem]
    print("RESULT " + json.dumps({
        "problem": problem,
        "time_ms": round(elapsed_ms, 2),
        "repeats": REPEATS[problem],
        "solutions": solution_nb,
        "statistics": {k: int(v) for k, v in statistics.items()},
        "nucs_file": nucs.__file__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))
