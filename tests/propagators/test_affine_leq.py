import numpy as np

from ncs.propagators.propagators import ALG_AFFINE_LEQ, compute_domains, get_triggers
from ncs.utils import init_data_by_values, init_domains_by_values


class TestAffineLEQ:
    def test_get_triggers(self) -> None:
        data = init_data_by_values([8, 1, -1])
        assert np.all(get_triggers(ALG_AFFINE_LEQ, 2, data) == np.array([[True, False], [False, True]]))

    def test_compute_domains_1(self) -> None:
        domains = init_domains_by_values([(1, 10), (1, 10)])
        data = init_data_by_values([-1, 1, -1])
        assert np.all(compute_domains(ALG_AFFINE_LEQ, domains, data) == np.array([[1, 9], [2, 10]]))

    def test_compute_domains_2(self) -> None:
        domains = init_domains_by_values([(1, 10), (1, 10)])
        data = init_data_by_values([8, 1, 1])
        assert np.all(compute_domains(ALG_AFFINE_LEQ, domains, data) == np.array([[1, 7], [1, 7]]))
