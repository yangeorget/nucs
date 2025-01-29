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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

from nucs.constants import PROP_CONSISTENCY
from nucs.propagators.alldifferent_propagator import compute_domains_alldifferent, path_max, path_min, path_set
from tests.propagators.propagator_test import PropagatorTest


class TestAlldifferent(PropagatorTest):
    def test_path_min(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        assert path_min(a, 0) == 0
        assert path_min(a, 1) == 1
        assert path_min(a, 2) == 2
        assert path_min(a, 3) == 0
        assert path_min(a, 4) == 1

    def test_path_max(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        assert path_max(a, 0) == 4
        assert path_max(a, 1) == 3
        assert path_max(a, 2) == 4
        assert path_max(a, 3) == 3
        assert path_max(a, 4) == 4

    def test_path_set(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        path_set(a, 0, 4, -1)
        assert np.all(a == np.array([-1, 3, -1, 0, 1]))

    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)],
                [],
                PROP_CONSISTENCY,
                [[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]],
            ),
            ([(0, 0), (2, 2), (1, 2)], [], PROP_CONSISTENCY, [[0, 0], [2, 2], [1, 1]]),
            ([(0, 0), (0, 4), (0, 4), (0, 4), (0, 4)], [], PROP_CONSISTENCY, [[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]]),
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
            compute_domains_alldifferent, domains, parameters, consistency_result, expected_domains
        )
