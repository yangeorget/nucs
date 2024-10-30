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
from nucs.propagators.gcc_propagator import compute_domains_gcc


class TestGCC:
    def test_compute_domains_0(self) -> None:
        domains = new_shr_domains_by_values([0])
        assert compute_domains_gcc(domains, new_parameters_by_values([0, 1, 1])) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0]]))

    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([0, 1])
        assert compute_domains_gcc(domains, new_parameters_by_values([0, 1, 1, 1, 1])) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([0, (0, 1)])
        assert compute_domains_gcc(domains, new_parameters_by_values([0, 1, 1, 1, 1])) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([0, 2, (1, 2)])
        assert compute_domains_gcc(domains, new_parameters_by_values([0] + [1] * 6)) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [2, 2], [1, 1]]))

    def test_compute_domains_4(self) -> None:
        domains = new_shr_domains_by_values([0, (0, 4), (0, 4), (0, 4), (0, 4)])
        assert compute_domains_gcc(domains, new_parameters_by_values([0] + [1] * 10)) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]]))

    def test_compute_domains_5(self) -> None:
        domains = new_shr_domains_by_values([(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)])
        assert compute_domains_gcc(domains, new_parameters_by_values([1] + [1] * 12)) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]]))

    def test_compute_domains_6(self) -> None:
        domains = new_shr_domains_by_values([(3, 4), (2, 4), (3, 4), (2, 5), (3, 6), (1, 6)])
        assert compute_domains_gcc(domains, new_parameters_by_values([1] + [0] * 6 + [1] * 6)) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[3, 4], [2, 2], [3, 4], [5, 5], [6, 6], [1, 1]]))
