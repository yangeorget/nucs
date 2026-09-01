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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.element_l_eq_alldifferent_propagator import compute_domains_element_l_eq_alldifferent
from tests.propagators.propagator_test import PropagatorTest


class TestElementLEqAlldifferent(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # v is fixed, exactly one l[idx] is the matching singleton -> i is set, entailment
            ([5, 3, 7, (0, 2), 3], [], PROP_ENTAILMENT, [[5, 5], [3, 3], [7, 7], [1, 1], [3, 3]]),
            # v is fixed, all l[idx] disjoint from v -> i becomes empty, inconsistency
            ([5, 6, 7, (0, 2), 3], [], PROP_INCONSISTENCY, None),
            # v is fixed, some l[idx] disjoint at the end -> i[DOMAIN_MAX] tightened
            ([(1, 5), (10, 15), (0, 1), 3], [], PROP_ENTAILMENT, [[3, 3], [10, 15], [0, 0], [3, 3]]),
            # v is fixed, some l[idx] disjoint at the start -> i[DOMAIN_MIN] increased
            ([(10, 15), (1, 5), (0, 1), 3], [], PROP_ENTAILMENT, [[10, 15], [3, 3], [1, 1], [3, 3]]),
            # v not fixed, multiple intersecting l[idx] -> v narrowed by l_v_min / l_v_max
            ([5, (4, 6), (0, 1), (3, 8)], [], PROP_CONSISTENCY, [[5, 5], [4, 6], [0, 1], [4, 6]]),
            # v not fixed, i becomes fixed (only one matches), l[i] = v but v still not fixed
            ([(1, 5), (10, 15), (0, 1), (3, 4)], [], PROP_CONSISTENCY, [[3, 4], [10, 15], [0, 0], [3, 4]]),
            # v not fixed, all l disjoint -> inconsistency
            ([(10, 15), (20, 25), (0, 1), (1, 5)], [], PROP_INCONSISTENCY, None),
            # i given out of bounds gets clamped
            ([(1, 5), (2, 6), (-5, 10), (3, 4)], [], PROP_CONSISTENCY, [[1, 5], [2, 6], [0, 1], [3, 4]]),
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
            compute_domains_element_l_eq_alldifferent, domains, parameters, consistency_result, expected_domains
        )
