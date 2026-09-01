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

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.member_reif_propagator import compute_domains_member_reif
from tests.propagators.propagator_test import PropagatorTest

# (b, x) domains and allowed values, for the cases that filter or decide something
FILTERING_CASES = [
    # b unfixed: neither outcome is settled, so nothing is filtered
    ([(0, 1), (0, 10)], [2, 5, 7], PROP_CONSISTENCY, [[0, 1], [0, 10]]),
    # no allowed value in range -> b is false, and the constraint can never be violated again
    ([(0, 1), (3, 4)], [2, 5, 9], PROP_ENTAILMENT, [[0, 0], [3, 4]]),
    # x's whole interval is allowed -> b is true
    ([(0, 1), (2, 4)], [1, 2, 3, 4, 5], PROP_ENTAILMENT, [[1, 1], [2, 4]]),
    # the empty set: x is never a member
    ([(0, 1), (0, 10)], [], PROP_ENTAILMENT, [[0, 0], [0, 10]]),
    # b true: the bounds snap onto the outermost allowed values, the hole at 6 is left in place
    ([1, (0, 10)], [2, 5, 7], PROP_CONSISTENCY, [[1, 1], [2, 7]]),
    # b true and the allowed values in range are consecutive -> entailment
    ([1, (0, 10)], [3, 4, 5], PROP_ENTAILMENT, [[1, 1], [3, 5]]),
    # b true and a single allowed value is left
    ([1, (3, 7)], [1, 5, 9], PROP_ENTAILMENT, [[1, 1], [5, 5]]),
    # b true but no allowed value in range
    ([1, (3, 4)], [2, 5, 9], PROP_INCONSISTENCY, None),
    # b false: each bound steps past the run of allowed values it sits on (2 then 3 at the bottom)
    ([0, (2, 10)], [2, 3, 7], PROP_CONSISTENCY, [[0, 0], [4, 10]]),
    # b false: the upper bound steps down past the trailing run
    ([0, (0, 9)], [4, 8, 9], PROP_CONSISTENCY, [[0, 0], [0, 7]]),
    # b false and the interior hole is unreachable for interval domains -> no change, still not entailed
    ([0, (0, 10)], [5], PROP_CONSISTENCY, [[0, 0], [0, 10]]),
    # b false and nothing in range is allowed -> entailment
    ([0, (3, 4)], [2, 5, 9], PROP_ENTAILMENT, [[0, 0], [3, 4]]),
    # b false but every value of x is allowed -> inconsistency
    ([0, (2, 3)], [2, 3], PROP_INCONSISTENCY, None),
    # b false and x bound to an allowed value -> inconsistency
    ([0, 5], [2, 5, 9], PROP_INCONSISTENCY, None),
    # b true and x bound to an allowed value -> entailment
    ([1, 5], [2, 5, 9], PROP_ENTAILMENT, [[1, 1], [5, 5]]),
]


class TestMemberReif(PropagatorTest):
    @pytest.mark.parametrize("domains,parameters,consistency_result,expected_domains", FILTERING_CASES)
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(
            compute_domains_member_reif, domains, parameters, consistency_result, expected_domains
        )

    @pytest.mark.parametrize("domains,parameters,consistency_result,expected_domains", FILTERING_CASES)
    def test_idempotence(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        """A single call reaches the propagator's fixpoint: feeding its result back changes nothing.

        The engine never reschedules a propagator onto itself after it prunes, so anything left on the table
        by the first call would simply be lost.
        """
        if expected_domains is None:
            return
        self.assert_compute_domains(
            compute_domains_member_reif,
            [(domain[0], domain[1]) for domain in expected_domains],
            parameters,
            consistency_result,
            expected_domains,
        )

    @pytest.mark.parametrize(
        "values",
        [[], [0], [2], [0, 3], [1, 2], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]],
    )
    def test_soundness_against_brute_force(self, values: list[int]) -> None:
        """Over every small (b, x) domain pair, the propagator never prunes a value that belongs to a
        solution, never keeps a state that has none, and only claims entailment when every remaining
        assignment satisfies the constraint."""
        universe = range(-1, 5)
        for b_min, b_max in [(0, 0), (1, 1), (0, 1)]:
            for x_min, x_max in itertools.combinations_with_replacement(universe, 2):
                solutions = [
                    (b, x)
                    for b in range(b_min, b_max + 1)
                    for x in range(x_min, x_max + 1)
                    if b == (1 if x in values else 0)
                ]
                domains = np.array([(b_min, b_max), (x_min, x_max)], dtype=np.int32)
                status = compute_domains_member_reif(domains, np.array(values, dtype=np.int32))
                if not solutions:
                    assert status == PROP_INCONSISTENCY, f"{values} {b_min}..{b_max} {x_min}..{x_max}"
                    continue
                assert status != PROP_INCONSISTENCY, f"{values} {b_min}..{b_max} {x_min}..{x_max}"
                # every solution of the original state survives the filtering
                for b, x in solutions:
                    assert domains[0, DOMAIN_MIN] <= b <= domains[0, DOMAIN_MAX]
                    assert domains[1, DOMAIN_MIN] <= x <= domains[1, DOMAIN_MAX]
                # the bounds it kept are the tightest an interval domain can express
                assert domains[0, DOMAIN_MIN] == min(b for b, _ in solutions)
                assert domains[0, DOMAIN_MAX] == max(b for b, _ in solutions)
                assert domains[1, DOMAIN_MIN] == min(x for _, x in solutions)
                assert domains[1, DOMAIN_MAX] == max(x for _, x in solutions)
                if status == PROP_ENTAILMENT:
                    # entailed means no assignment left in the filtered box can violate the constraint
                    box = [
                        (b, x)
                        for b in range(domains[0, DOMAIN_MIN], domains[0, DOMAIN_MAX] + 1)
                        for x in range(domains[1, DOMAIN_MIN], domains[1, DOMAIN_MAX] + 1)
                    ]
                    assert all(b == (1 if x in values else 0) for b, x in box)
