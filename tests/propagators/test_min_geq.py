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
from nucs.propagators.min_geq_propagator import compute_domains_min_geq
from tests.propagators.propagator_test import PropagatorTest


class TestMinGeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 4), (2, 5), (2, 6)], [], PROP_CONSISTENCY, [[2, 4], [2, 5], [2, 4]]),
            ([(1, 3), (3, 3), (4, 5)], [], PROP_INCONSISTENCY, None),
            ([(2, 4), (2, 5), (6, 8)], [], PROP_INCONSISTENCY, None),
            ([(2, 3), (2, 3), (0, 1)], [], PROP_ENTAILMENT, [[2, 3], [2, 3], [0, 1]]),
            ([(0, 1), (0, 1), (1, 1)], [], PROP_CONSISTENCY, [[1, 1], [1, 1], [1, 1]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_min_geq, domains, parameters, consistency_result, expected_domains)
