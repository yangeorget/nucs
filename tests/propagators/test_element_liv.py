import numpy as np

from nucs.memory import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY, new_data_by_values, new_domains_by_values
from nucs.propagators.element_liv_propagator import compute_domains_element_liv


class TestElementLIV:
    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(-1, 0), (1, 2), (0, 2), (-1, 1)])
        data = new_data_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[-1, 0], [1, 2], [0, 1], [-1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(-4, -2), (1, 2), (0, 1), (0, 1)])
        data = new_data_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[-4, -2], [1, 1], [1, 1], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(-4, -2), (1, 2), (0, 1), (0, 0)])
        data = new_data_by_values([])
        assert compute_domains_element_liv(domains, data) == PROP_INCONSISTENCY
