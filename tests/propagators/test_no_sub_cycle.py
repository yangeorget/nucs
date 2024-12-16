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
from nucs.propagators.no_sub_cycle_propagator import compute_domains_no_sub_cycle


class TestNoSubCycle:
    def test_compute_domains_3_consistency(self) -> None:
        domains = new_shr_domains_by_values([(1, 1), (2, 2), (0, 0)])
        data = new_parameters_by_values([])
        assert compute_domains_no_sub_cycle(domains, data) == PROP_CONSISTENCY

    def test_compute_domains_3_inconsistency(self) -> None:
        domains = new_shr_domains_by_values([(1, 1), (0, 0), (2, 2)])
        data = new_parameters_by_values([])
        assert compute_domains_no_sub_cycle(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_single(self) -> None:
        domains = new_shr_domains_by_values([0])
        data = new_parameters_by_values([])
        assert compute_domains_no_sub_cycle(domains, data) == PROP_INCONSISTENCY
