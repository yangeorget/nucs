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

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.permutation_aux_propagator import compute_domains_permutation_aux
from tests.propagators.propagator_test import PropagatorTest


class TestPermutationAux(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # all bound, valid permutation -> consistency, unchanged
            ([1, 0, 1, 0], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # prev all bound, propagates to next (prev[0]=1 -> next[1]=0; prev[1]=0 -> next[0]=1)
            ([(0, 1), (0, 1), 1, 0], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # next all bound, propagates to prev (by symmetry)
            ([1, 0, (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # both next pointing to 0 -> inconsistent
            ([0, 0, (0, 1), (0, 1)], [], PROP_INCONSISTENCY, None),
            # partial filtering: next[0]=0 forces prev[0]=0 and prev[1]=1
            ([0, (0, 1), (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 0], [1, 1], [0, 0], [1, 1]]),
            # disjoint prefix raises prev[j,MIN]; disjoint suffix lowers prev[j,MAX]
            (
                [2, 1, 0, (0, 2), (0, 2), (0, 2)],
                [],
                PROP_CONSISTENCY,
                [[2, 2], [1, 1], [0, 0], [2, 2], [1, 1], [0, 0]],
            ),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(
            compute_domains_permutation_aux, domains, parameters, consistency_result, expected_domains
        )
