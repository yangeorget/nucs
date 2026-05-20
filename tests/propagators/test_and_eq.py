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
from nucs.propagators.and_eq_propagator import compute_domains_and_eq
from tests.propagators.propagator_test import PropagatorTest


class TestAndEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # some x is 0 and y was free -> y forced to 0, entailed
            ([(0, 1), 0, (0, 1), (0, 1)], [], PROP_ENTAILMENT, [[0, 1], [0, 0], [0, 1], [0, 0]]),
            # some x is 0 but y is fixed to 1 -> inconsistent
            ([(0, 1), 0, (0, 1), 1], [], PROP_INCONSISTENCY, None),
            # all x are 1 and y was free -> y forced to 1, entailed
            ([1, 1, 1, (0, 1)], [], PROP_ENTAILMENT, [[1, 1], [1, 1], [1, 1], [1, 1]]),
            # all x are 1 but y is fixed to 0 -> inconsistent
            ([1, 1, 1, 0], [], PROP_INCONSISTENCY, None),
            # y is fixed to 1 -> all x forced to 1, entailed
            ([(0, 1), (0, 1), (0, 1), 1], [], PROP_ENTAILMENT, [[1, 1], [1, 1], [1, 1], [1, 1]]),
            # y is fixed to 0 and exactly one x can be 0 -> that x forced to 0, entailed
            ([1, (0, 1), 1, 0], [], PROP_ENTAILMENT, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # y is fixed to 0 and several x's can still be 0 -> consistent only
            ([(0, 1), (0, 1), 1, 0], [], PROP_CONSISTENCY, [[0, 1], [0, 1], [1, 1], [0, 0]]),
            # nothing can be deduced
            ([(0, 1), (0, 1), (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 1], [0, 1], [0, 1], [0, 1]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_and_eq, domains, parameters, consistency_result, expected_domains)
