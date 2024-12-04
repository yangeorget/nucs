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
from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.propagators.scc_propagator import compute_domains_scc


class TestSCC:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(0, 2), (0, 2), (0, 2)])
        data = new_parameters_by_values([])
        assert compute_domains_scc(domains, data) == PROP_CONSISTENCY

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(2, 2), (0, 0), (1, 1)])
        data = new_parameters_by_values([])
        assert compute_domains_scc(domains, data) == PROP_CONSISTENCY

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(1, 1), (0, 0), (2, 2)])
        data = new_parameters_by_values([])
        assert compute_domains_scc(domains, data) == PROP_INCONSISTENCY
