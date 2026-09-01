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
import random

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.cumulative_propagator import (
    compute_domains_cumulative,
    is_vacuous_cumulative,
    is_vacuous_cumulative_var,
)
from nucs.propagators.propagators import ALG_CUMULATIVE, ALG_CUMULATIVE_VAR
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest


def _feasible_starts(
    bounds: list[tuple[int, int]], durations: list[int], heights: list[int], capacity: int
) -> list[tuple[int, ...]]:
    """Brute-force every assignment of start times within bounds whose resource profile fits under capacity."""
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for starts in itertools.product(*ranges):
        horizon = max(starts[i] + durations[i] for i in range(len(starts)))
        ok = True
        for t in range(horizon):
            load = sum(heights[i] for i in range(len(starts)) if starts[i] <= t < starts[i] + durations[i])
            if load > capacity:
                ok = False
                break
        if ok:
            feasible.append(starts)
    return feasible


class TestCumulative(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # capacity 2, two unit-height tasks with slack: no compulsory part, no pruning
            ([(0, 10), (0, 10)], [2, 2, 1, 1, 2], PROP_CONSISTENCY, [[0, 10], [0, 10]]),
            # a fixed task occupies [2, 5) at height 1 on a capacity-1 resource: the second task is pushed past it
            ([(2, 2), (1, 10)], [3, 2, 1, 1, 1], PROP_CONSISTENCY, [[2, 2], [5, 10]]),
            # variable heights: a fixed height-2 task fills [0, 4) of a capacity-3 resource, so a height-2 task
            # (2 + 2 > 3) cannot overlap it and is pushed to 4
            ([(0, 0), (0, 10)], [4, 2, 2, 2, 3], PROP_CONSISTENCY, [[0, 0], [4, 10]]),
            # overload: two fixed unit tasks both run during [1, 3) on a capacity-1 resource
            ([(0, 0), (1, 1)], [3, 3, 1, 1, 1], PROP_INCONSISTENCY, None),
            # overload with heights: two fixed height-2 tasks overlap on a capacity-3 resource (2 + 2 > 3)
            ([(0, 0), (0, 0)], [2, 2, 2, 2, 3], PROP_INCONSISTENCY, None),
            # a single task whose demand exceeds the capacity can never run
            ([(0, 5)], [2, 3, 2], PROP_INCONSISTENCY, None),
            # all starts fixed and the profile fits: entailed
            ([(0, 0), (3, 3)], [3, 3, 1, 1, 1], PROP_ENTAILMENT, [[0, 0], [3, 3]]),
            # two height-1 tasks may overlap under capacity 2: fixed and consistent, entailed
            ([(0, 0), (0, 0)], [2, 2, 1, 1, 2], PROP_ENTAILMENT, [[0, 0], [0, 0]]),
            # energetic reasoning, beyond timetabling: neither task has a compulsory part, but over [3, 6) the
            # short task needs 1 of the 3 capacity-1 units, leaving room for only 2 of the long task's 3, so
            # its earliest start is pushed from 3 to 4 (timetabling alone prunes nothing here)
            ([(3, 8), (3, 4)], [3, 1, 1, 1, 1], PROP_CONSISTENCY, [[4, 8], [3, 4]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(
            compute_domains_cumulative, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # for many small random instances the propagator must be sound: never remove a start that belongs to a
        # feasible schedule, and never claim inconsistency when a feasible schedule exists. (timetabling is
        # incomplete, so it may stay consistent on an infeasible instance -- that is allowed.)
        rng = random.Random(20260622)
        for _ in range(5000):
            n = rng.randint(2, 4)
            capacity = rng.randint(1, 4)
            durations = [rng.randint(1, 3) for _ in range(n)]
            heights = [rng.randint(1, 3) for _ in range(n)]
            bounds = []
            for _ in range(n):
                lo = rng.randint(0, 4)
                hi = lo + rng.randint(0, 4)
                bounds.append((lo, hi))
            feasible = _feasible_starts(bounds, durations, heights, capacity)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            parameters = np.array(durations + heights + [capacity], dtype=np.int32)
            result = compute_domains_cumulative(domains, parameters)
            if result == PROP_INCONSISTENCY:
                assert not feasible, (
                    f"declared inconsistent but feasible: {bounds} p={durations} h={heights} c={capacity}"
                )
                continue
            if not feasible:
                continue  # timetabling is incomplete: staying consistent on an infeasible instance is sound
            for i in range(n):
                bc_min = min(s[i] for s in feasible)
                bc_max = max(s[i] for s in feasible)
                # soundness: the filtered interval must keep every feasible value
                assert domains[i, DOMAIN_MIN] <= bc_min, (
                    f"over-pruned MIN of {i}: {bounds} p={durations} h={heights} c={capacity}"
                )
                assert domains[i, DOMAIN_MAX] >= bc_max, (
                    f"over-pruned MAX of {i}: {bounds} p={durations} h={heights} c={capacity}"
                )

    @pytest.mark.parametrize(
        "n,parameters",
        [
            (3, [2, 2, 2, 1, 1, 1, 5]),  # demands sum to 3, capacity 5: never exceeded
            (3, [1, 2, 3, 2, 2, 2, 6]),  # demands sum exactly to the capacity
            (3, [0, 0, 0, 9, 9, 9, 0]),  # no task ever occupies an instant, and the capacity is non-negative
        ],
    )
    def test_a_vacuous_cumulative_is_not_posted(self, n: int, parameters: list[int]) -> None:
        """A capacity that already covers the sum of every demand cannot be exceeded, whatever the starts,
        so the constraint is settled by its parameters alone and never becomes a propagator."""
        assert is_vacuous_cumulative(n, parameters, [(0, 4)] * n)
        problem = Problem([(0, 4)] * n)
        problem.add_propagator(ALG_CUMULATIVE, range(n), parameters)
        assert problem.propagator_nb == 0

    @pytest.mark.parametrize(
        "n,parameters",
        [
            (3, [2, 2, 2, 3, 3, 3, 5]),  # demands sum to 9 above a capacity of 5: the constraint binds
            (3, [0, 0, 0, 9, 9, 9, -1]),  # zero durations, but a negative capacity is violated by a usage of 0
        ],
    )
    def test_a_binding_cumulative_is_posted(self, n: int, parameters: list[int]) -> None:
        assert not is_vacuous_cumulative(n, parameters, [(0, 4)] * n)
        problem = Problem([(0, 4)] * n)
        problem.add_propagator(ALG_CUMULATIVE, range(n), parameters)
        assert problem.propagator_nb == 1

    def test_a_vacuous_cumulative_var_is_not_posted(self) -> None:
        parameters = [1, 1, 5]  # demands sum to 2, capacity 5
        assert is_vacuous_cumulative_var(4, parameters, [(0, 3), (0, 3), (0, 2), (0, 2)])
        problem = Problem([(0, 3), (0, 3), (0, 2), (0, 2)])
        problem.add_propagator(ALG_CUMULATIVE_VAR, range(4), parameters)
        assert problem.propagator_nb == 0

    def test_a_binding_cumulative_var_is_posted(self) -> None:
        parameters = [3, 3, 5]  # demands sum to 6 above a capacity of 5
        assert not is_vacuous_cumulative_var(4, parameters, [(0, 3), (0, 3), (0, 2), (0, 2)])
        problem = Problem([(0, 3), (0, 3), (0, 2), (0, 2)])
        problem.add_propagator(ALG_CUMULATIVE_VAR, range(4), parameters)
        assert problem.propagator_nb == 1

    @pytest.mark.parametrize(
        "alg,n,parameters,domains",
        [
            (ALG_CUMULATIVE, 3, [2, 2, 2, 1, 1, 1, 5], [(0, 4)] * 3),
            (ALG_CUMULATIVE, 3, [0, 0, 0, 9, 9, 9, 0], [(0, 3)] * 3),
            (ALG_CUMULATIVE_VAR, 4, [1, 1, 5], [(0, 3), (0, 3), (0, 2), (0, 2)]),
        ],
    )
    def test_dropping_a_vacuous_propagator_preserves_the_solutions(
        self, alg: int, n: int, parameters: list[int], domains: list[tuple[int, int]]
    ) -> None:
        """Not posting the propagator must leave exactly the solutions that running it would have left.

        This is the property the whole mechanism rests on, and it is what ruled a disjunctive guard out:
        NuCS's disjunctive is the strict one, where a zero-duration task still may not fall inside another.
        """
        dropped = Problem(list(domains))
        dropped.add_propagator(alg, range(n), parameters)
        assert dropped.propagator_nb == 0
        posted = Problem(list(domains))  # bypass the check so the propagator really runs
        posted.propagators.append((list(range(n)), alg, list(parameters)))
        posted.propagator_nb += 1
        assert sorted(tuple(s) for s in BacktrackSolver(dropped, log_level="ERROR").find_all()) == sorted(
            tuple(s) for s in BacktrackSolver(posted, log_level="ERROR").find_all()
        )
