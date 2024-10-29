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
from nucs.propagators.element_liv_propagator import compute_domains_element_liv


class TestElementLIV:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(-1, 0), (1, 2), (0, 2), (-1, 1)])
        data = new_parameters_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[-1, 0], [1, 2], [0, 1], [-1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(-4, -2), (1, 2), (0, 1), (0, 1)])
        data = new_parameters_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[-4, -2], [1, 1], [1, 1], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(-4, -2), (1, 2), (0, 1), (0, 0)])
        data = new_parameters_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_INCONSISTENCY
