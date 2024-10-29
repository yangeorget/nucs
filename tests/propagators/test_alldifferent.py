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
# Copyright 2024 - Yan Georget
###############################################################################
import numpy as np

from nucs.constants import PROP_CONSISTENCY
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.propagators.alldifferent_propagator import compute_domains_alldifferent, path_max, path_min, path_set


class TestAlldifferent:
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

    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)])
        data = new_parameters_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(0, 0), (2, 2), (1, 2)])
        data = new_parameters_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [2, 2], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(0, 0), (0, 4), (0, 4), (0, 4), (0, 4)])
        data = new_parameters_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]]))
