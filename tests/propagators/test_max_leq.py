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
from nucs.propagators.max_leq_propagator import compute_domains_max_leq
from tests.propagators.propagator_test import PropagatorTest


class TestMaxLeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 4), (2, 5), (0, 2)], [], PROP_CONSISTENCY, [[1, 2], [2, 2], [2, 2]]),
            ([(1, 4), (3, 5), (0, 2)], [], PROP_INCONSISTENCY, None),
            ([(2, 4), (2, 5), (0, 1)], [], PROP_INCONSISTENCY, None),
            ([(0, 1), (0, 1), (2, 3)], [], PROP_ENTAILMENT, [[0, 1], [0, 1], [2, 3]]),
            ([(0, 1), (0, 1), (0, 0)], [], PROP_CONSISTENCY, [[0, 0], [0, 0], [0, 0]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_max_leq, domains, parameters, consistency_result, expected_domains)
