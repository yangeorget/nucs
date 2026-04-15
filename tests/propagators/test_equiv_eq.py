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
from typing import List, Optional, Tuple, Union

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.equiv_eq_propagator import compute_domains_equiv_eq
from tests.propagators.propagator_test import PropagatorTest


class TestEquivEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # b is 1, so x = y: intersection
            ([(1, 1), (3, 5), (4, 6)], [], PROP_CONSISTENCY, [[1, 1], [4, 5], [4, 5]]),
            # b is 1, x = y already
            ([(1, 1), (4, 4), (4, 4)], [], PROP_ENTAILMENT, [[1, 1], [4, 4], [4, 4]]),
            # b is 1, no overlap
            ([(1, 1), (3, 4), (5, 6)], [], PROP_INCONSISTENCY, None),
            # b is 0, so x != y: already different
            ([(0, 0), (3, 4), (5, 6)], [], PROP_ENTAILMENT, [[0, 0], [3, 4], [5, 6]]),
            # b unknown, x and y have no overlap -> b = 0
            ([(0, 1), (3, 4), (5, 6)], [], PROP_ENTAILMENT, [[0, 0], [3, 4], [5, 6]]),
            # b unknown, x and y fixed same -> b = 1
            ([(0, 1), (4, 4), (4, 4)], [], PROP_ENTAILMENT, [[1, 1], [4, 4], [4, 4]]),
            # b unknown, x and y fixed different -> b = 0
            ([(0, 1), (4, 4), (5, 5)], [], PROP_ENTAILMENT, [[0, 0], [4, 4], [5, 5]]),
            # b unknown, no info
            ([(0, 1), (3, 5), (4, 6)], [], PROP_CONSISTENCY, [[0, 1], [3, 5], [4, 6]]),
            # b is 0, x and y both fixed same (inconsistent)
            ([(0, 0), (4, 4), (4, 4)], [], PROP_INCONSISTENCY, None),
            # b is 1, x and y both fixed different (inconsistent)
            ([(1, 1), (4, 4), (5, 5)], [], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_equiv_eq, domains, parameters, consistency_result, expected_domains)
