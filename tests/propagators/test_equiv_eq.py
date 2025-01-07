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
import numpy as np

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.propagators.equiv_eq_propagator import compute_domains_equiv_eq


class TestEquivEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), (3, 5)])
        data = new_parameters_by_values([4])
        assert compute_domains_equiv_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 1], [3, 5]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(1, 1), (3, 5)])
        data = new_parameters_by_values([4])
        assert compute_domains_equiv_eq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[1, 1], [4, 4]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), (3, 5)])
        data = new_parameters_by_values([6])
        assert compute_domains_equiv_eq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[0, 0], [3, 5]]))

    def test_compute_domains_4(self) -> None:
        domains = new_shr_domains_by_values([(1, 1), (3, 5)])
        data = new_parameters_by_values([6])
        assert compute_domains_equiv_eq(domains, data) == PROP_INCONSISTENCY
