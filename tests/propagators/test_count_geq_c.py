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
from nucs.propagators.count_geq_c_propagator import compute_domains_count_geq_c
from tests.propagators.propagator_test import PropagatorTest


class TestCountGeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5],
                [9, 1],
                PROP_INCONSISTENCY,
                None,
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5],
                [5, 2],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5]],
            ),
            (
                [(1, 4), (3, 5), (3, 5), (6, 8), 3, 5],
                [5, 3],
                PROP_ENTAILMENT,
                [[1, 4], [5, 5], [5, 5], [6, 8], [3, 3], [5, 5]],
            ),
            (
                [(1, 3), (1, 3), (5, 7), (5, 7)],
                [2, 2],
                PROP_ENTAILMENT,
                [[2, 2], [2, 2], [5, 7], [5, 7]],
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
        self.assert_compute_domains(
            compute_domains_count_geq_c, domains, parameters, consistency_result, expected_domains
        )
