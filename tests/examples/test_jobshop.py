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
import json

import pytest

from nucs.examples.jobshop.jobshop_problem import JobShopProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


def _assert_valid_schedule(problem: JobShopProblem, solution) -> int:  # type: ignore[no-untyped-def]
    """Checks that a solution respects job precedences and machine non-overlap, and returns the makespan."""
    starts = [[int(solution[problem.start(j, k)]) for k in range(problem.machine_nb)] for j in range(problem.job_nb)]
    completion = 0
    machine_intervals: dict = {}
    for j, job in enumerate(problem.jobs):
        for k, (machine, duration) in enumerate(job):
            s = starts[j][k]
            if k > 0:  # job precedence: operation k starts after operation k-1 completes
                prev_machine, prev_duration = job[k - 1]
                assert starts[j][k - 1] + prev_duration <= s
            machine_intervals.setdefault(machine, []).append((s, s + duration))
            completion = max(completion, s + duration)
    for intervals in machine_intervals.values():  # machine non-overlap
        intervals.sort()
        for (_, end), (start, _) in zip(intervals, intervals[1:]):
            assert end <= start
    return completion


class TestJobShop:
    @pytest.mark.parametrize(
        "name, optimum",
        [("mt06", 55), ("mt20", 1165), ("la01", 666), ("la31", 1784)],
    )
    def test_jobshop_optimality_proof(self, name: str, optimum: int) -> None:
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
        ],
    )
    def test_jobshop_first_solution(self, name: str, optimum: int) -> None:
        # the critical-resource search keeps branching on one machine until its tasks are sequenced, then
        # schedules each task at its earliest start: a valid schedule whose first-solution makespan is
        # reproducible (and, on the larger instances, far better than the smallest-domain baseline)
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        problem = JobShopProblem(dataset["jobs"])
        solution = next(BacktrackSolver(problem, searches=problem.recommended_searches()).solve(), None)
        assert solution is not None
        makespan = _assert_valid_schedule(problem, solution)
        assert makespan == int(solution[problem.makespan])
        assert makespan >= optimum

    @pytest.mark.parametrize(
        "name, optimum",
        [("mt06", 55)],
    )
    def test_jobshop_best_solution(self, name: str, optimum: int) -> None:
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        assert dataset["optimum"] == optimum
        problem = JobShopProblem(dataset["jobs"])
        solver = BacktrackSolver(problem, searches=problem.recommended_searches())
        solution = solver.minimize(problem.makespan)
        assert solution is not None
        assert solution[problem.makespan] == optimum
