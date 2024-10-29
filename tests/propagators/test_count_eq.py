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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.propagators.count_eq_propagator import compute_domains_count_eq


class TestCountEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 1])
        data = new_parameters_by_values([5])
        assert compute_domains_count_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 2])
        data = new_parameters_by_values([5])
        assert compute_domains_count_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 0])
        data = new_parameters_by_values([5])
        assert compute_domains_count_eq(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), 5, (3, 6), (6, 8), 3, 5, (1, 2)])
        data = new_parameters_by_values([5])
        assert compute_domains_count_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [5, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]]))

    def test_compute_domains_5(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, (-1, 10)])
        data = new_parameters_by_values([5])
        assert compute_domains_count_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [1, 3]]))

    def test_compute_domains_6(self) -> None:
        domains = new_shr_domains_by_values([2, (0, 1), (3, 4), 2, 2, (2, 4)])
        data = new_parameters_by_values([2])
        assert compute_domains_count_eq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[2, 2], [0, 1], [3, 4], [2, 2], [2, 2], [3, 3]]))
