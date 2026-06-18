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

from nucs.constants import OPTIM_PRUNE
from nucs.examples.jobshop.jobshop_problem import JobShopProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
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
    def test_jobshop_optimum_small(self) -> None:
        # a 2-job, 2-machine instance whose optimum makespan is 5 (provable quickly)
        jobs = [[[0, 2], [1, 3]], [[1, 2], [0, 1]]]
        problem = JobShopProblem(jobs)
        solver = BacktrackSolver(
            problem, decision_variables=range(problem.completion_start), var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN
        )
        solution = solver.minimize(problem.makespan, mode=OPTIM_PRUNE)
        assert solution is not None
        assert _assert_valid_schedule(problem, solution) == 5
        assert int(solution[problem.makespan]) == 5

    # Proving the optima of these classic benchmarks is out of reach for a bounds branch-and-bound solver
    # (e.g. mt10 stayed open for decades), so we only check that the disjunctive model yields a valid
    # schedule whose makespan is no smaller than the known optimum.
    @pytest.mark.parametrize(
        "name, optimum",
        [
            ("mt06", 55),
            ("mt10", 930),
            ("mt20", 1165),
            ("la05", 593),
        ],
    )
    def test_jobshop_feasible(self, name: str, optimum: int) -> None:
        with open(f"datasets/examples/jobshop/{name}.json", "r") as json_file:
            dataset = json.load(json_file)
        assert dataset["optimum"] == optimum
        problem = JobShopProblem(dataset["jobs"])
        solver = BacktrackSolver(
            problem,
            decision_variables=range(problem.completion_start),
            var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
            log_level="ERROR",
        )
        solution = next(solver.solve(), None)
        assert solution is not None
        makespan = _assert_valid_schedule(problem, solution)
        assert makespan == int(solution[problem.makespan])
        assert makespan >= optimum
