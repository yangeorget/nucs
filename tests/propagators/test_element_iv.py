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
from nucs.propagators.element_iv_propagator import compute_domains_element_iv


class TestElementIV:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(0, 4), (-2, 2)])
        data = new_parameters_by_values([3, 0, 1, 2, 4])
        assert compute_domains_element_iv(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 3], [0, 2]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(0, 4), (-2, -1)])
        data = new_parameters_by_values([3, 0, 1, 2, 4])
        assert compute_domains_element_iv(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(0, 4), (-2, 0)])
        data = new_parameters_by_values([3, 0, 1, 2, 4])
        assert compute_domains_element_iv(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[1, 1], [0, 0]]))
