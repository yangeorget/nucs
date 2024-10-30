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
from nucs.propagators.relation_propagator import compute_domains_relation


class TestRelation:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(-5, 5), (-5, 5)])
        data = new_parameters_by_values([0, 7, 1, 4, 2, -7, 3, 3])
        assert compute_domains_relation(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 3], [3, 4]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(0, 3), (0, 3), (1, 8)])
        data = new_parameters_by_values(
            [0, 1, 0, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 2, 2, 1, 3, 3, 2, 1, 2, 2, 2, 4, 2, 3, 6, 3, 1, 3, 3, 2, 6, 3, 3, 9]
        )
        assert compute_domains_relation(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 3], [1, 3], [1, 6]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(0, 3), (0, 3)])
        data = new_parameters_by_values([4, 5])
        assert compute_domains_relation(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_shr_domains_by_values([(0, 3), (0, 3)])
        data = new_parameters_by_values([1, 2])
        assert compute_domains_relation(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[1, 1], [2, 2]]))

    def test_compute_domains_5(self) -> None:
        data = new_parameters_by_values([0, 1, 0, 0, 2, 1, 0, 3, 2, 1, 2, 3, 1, 3, 4, 2, 3, 5])
        assert compute_domains_relation(new_shr_domains_by_values([0, 1, (0, 5)]), data) == PROP_ENTAILMENT
        assert compute_domains_relation(new_shr_domains_by_values([0, 2, (0, 5)]), data) == PROP_ENTAILMENT
        assert compute_domains_relation(new_shr_domains_by_values([0, 3, (0, 5)]), data) == PROP_ENTAILMENT
        assert compute_domains_relation(new_shr_domains_by_values([2, 3, (0, 5)]), data) == PROP_ENTAILMENT
        assert compute_domains_relation(new_shr_domains_by_values([1, 3, (0, 5)]), data) == PROP_ENTAILMENT
        assert compute_domains_relation(new_shr_domains_by_values([1, 2, (0, 5)]), data) == PROP_ENTAILMENT
