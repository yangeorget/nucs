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
import itertools
import json

import pytest

from nucs.constants import DOMAIN_MIN
from nucs.examples.jobshop.jobshop_problem import JobShopProblem
from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import OPTIM_PRUNE


def _assert_valid_schedule(problem: JobShopProblem, solution) -> int:  # type: ignore[no-untyped-def]
    """Checks that a solution respects job precedences and machine non-overlap, and returns the makespan."""
    starts = [[int(solution[problem.start(j, k)]) for k in range(problem.machine_nb)] for j in range(problem.job_nb)]
    completion = 0
    machine_intervals: dict = {}
    for j, job in enumerate(problem.jobs):
        for k, (machine, duration) in enumerate(job):
            s = starts[j][k]
            if k > 0:  # job precedence: operation k starts after operation k-1 completes
                _prev_machine, prev_duration = job[k - 1]
                assert starts[j][k - 1] + prev_duration <= s
            machine_intervals.setdefault(machine, []).append((s, s + duration))
            completion = max(completion, s + duration)
    for intervals in machine_intervals.values():  # machine non-overlap
        intervals.sort()
        for (_, end), (start, _) in itertools.pairwise(intervals):
            assert end <= start
    return completion


class TestJobShop:
    @pytest.mark.parametrize(
        "name, optimum",
        [("mt06", 55), ("mt20", 1165), ("la01", 666), ("la31", 1784)],
    )
    def test_solve(self, name: str, optimum: int) -> None:
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        problem = JobShopProblem(dataset["jobs"], optimum - 1)
        solver = BacktrackSolver(problem, searches=problem.recommended_searches())
        solution = next(solver.solve(), None)
        assert solution is None

    @pytest.mark.parametrize(
        "name, optimum",
        [
            ("mt06", 55),
            ("mt10", 930),
            ("mt20", 1165),
            ("la01", 666),
            ("la05", 593),
            # without the JIT, la31 needs 3.2 s on a fast machine and 4.9 s under coverage on Python 3.12, which
            # traces each line: on the CI runner, more than the budget of 10 s; the other instances cover the same code
            pytest.param(
                "la31",
                1784,
                marks=pytest.mark.skipif(
                    bool(NUMBA_DISABLE_JIT), reason="300 choices of interpreted Python are too slow"
                ),
            ),
        ],
    )
    def test_solve_first_solution(self, name: str, optimum: int) -> None:
        # the Set Times search schedules the task that can start soonest at its earliest start: a valid schedule
        # whose first-solution makespan is reproducible; on la31, the critical-resource search found no schedule in
        # 30 s, where this one needs 300 choices
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        problem = JobShopProblem(dataset["jobs"])
        # the budget makes a search that finds no schedule fail instead of hanging
        solution = next(BacktrackSolver(problem, searches=problem.recommended_searches()).solve(timeout=10), None)
        assert solution is not None
        makespan = _assert_valid_schedule(problem, solution)
        assert makespan == int(solution[problem.makespan])
        assert makespan >= optimum

    # in the mode of the example, OPTIM_PRUNE, the Set Times search proves the optimum of la01 with 195 choices; the
    # critical-resource search does not in 10 s (it does in OPTIM_RESET, with 38,798 choices); the budget makes a
    # slower search fail instead of hanging
    @pytest.mark.parametrize(
        "name, optimum",
        [("mt06", 55), ("la01", 666)],
    )
    def test_find_best(self, name: str, optimum: int) -> None:
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        assert dataset["optimum"] == optimum
        problem = JobShopProblem(dataset["jobs"])
        solver = BacktrackSolver(problem, searches=problem.recommended_searches())
        solution = solver.find_best(problem.makespan, DOMAIN_MIN, mode=OPTIM_PRUNE, timeout=10)
        assert solution is not None
        assert solution[problem.makespan] == optimum
