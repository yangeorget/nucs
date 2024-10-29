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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.propagators.affine_leq_propagator import compute_domains_affine_leq, get_triggers_affine_leq


class TestAffineLEQ:
    def test_get_triggers(self) -> None:
        data = new_parameters_by_values([1, -1, 8])
        assert np.all(get_triggers_affine_leq(2, data) == np.array([[True, False], [False, True]]))

    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(1, 10), (1, 10)])
        data = new_parameters_by_values([1, -1, -1])
        assert compute_domains_affine_leq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 9], [2, 10]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(1, 10), (1, 10)])
        data = new_parameters_by_values([1, 1, 8])
        assert compute_domains_affine_leq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(2, 3), (1, 2)])
        data = new_parameters_by_values([1, 1, 5])
        assert compute_domains_affine_leq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[2, 3], [1, 2]]))
