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

import pytest

from nucs.constants import PROP_CONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.gcc_propagator import compute_domains_gcc, is_vacuous_gcc
from nucs.propagators.propagators import ALG_GCC
from tests.propagators.propagator_test import PropagatorTest


class TestGCC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([0], [0, 1, 1], PROP_CONSISTENCY, [[0, 0]]),
            ([0, 1], [0, 1, 1, 1, 1], PROP_CONSISTENCY, [[0, 0], [1, 1]]),
            ([0, (0, 1)], [0, 1, 1, 1, 1], PROP_CONSISTENCY, [[0, 0], [1, 1]]),
            ([0, 2, (1, 2)], [0] + [1] * 6, PROP_CONSISTENCY, [[0, 0], [2, 2], [1, 1]]),
            (
                [0, (0, 4), (0, 4), (0, 4), (0, 4)],
                [0] + [1] * 10,
                PROP_CONSISTENCY,
                [[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]],
            ),
            (
                [(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)],
                [1] + [1] * 12,
                PROP_CONSISTENCY,
                [[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]],
            ),
            (
                [(3, 4), (2, 4), (3, 4), (2, 5), (3, 6), (1, 6)],
                [1] + [0] * 6 + [1] * 6,
                PROP_CONSISTENCY,
                [[3, 4], [2, 2], [3, 4], [5, 5], [6, 6], [1, 1]],
            ),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_gcc, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize(
        "n,parameters",
        [
            # every value may be taken by all 3 variables and none is required: nothing to enforce
            (3, [0, 0, 0, 0, 3, 3, 3]),
            # upper capacities beyond the number of variables are just as free
            (2, [0, 0, 0, 7, 7]),
        ],
    )
    def test_a_vacuous_gcc_is_not_posted(self, n: int, parameters: list[int]) -> None:
        """A gcc that no assignment can violate is a property of the parameters, so it never becomes a
        propagator at all.

        MiniZinc emits these in quantity: global_cardinality_low_up leaves the values outside its cover
        unconstrained, so a cover whose own capacities do not bite makes the whole constraint vacuous.
        """
        assert is_vacuous_gcc(n, parameters, [(0, n - 1)] * n)
        problem = Problem([(0, n - 1)] * n)
        problem.add_propagator(ALG_GCC, range(n), parameters)
        assert problem.propagator_nb == 0

    @pytest.mark.parametrize(
        "n,parameters",
        [
            (3, [0, 0, 0, 0, 2, 2, 2]),  # an upper capacity below the number of variables still binds
            (3, [0, 1, 0, 0, 3, 3, 3]),  # a value required at least once still binds
        ],
    )
    def test_a_binding_gcc_is_posted(self, n: int, parameters: list[int]) -> None:
        assert not is_vacuous_gcc(n, parameters, [(0, n - 1)] * n)
        problem = Problem([(0, n - 1)] * n)
        problem.add_propagator(ALG_GCC, range(n), parameters)
        assert problem.propagator_nb == 1

    def test_a_binding_capacity_is_not_treated_as_vacuous(self) -> None:
        """An upper capacity below the number of variables still binds, so the guard must not fire: the
        constraint stays active and keeps filtering."""
        self.assert_compute_domains(
            compute_domains_gcc,
            [(0, 1), (0, 1), 0],  # at most 2 of the 3 variables per value, one already fixed to 0
            [0, 0, 0, 2, 2],
            PROP_CONSISTENCY,
            [[0, 1], [0, 1], [0, 0]],
        )
