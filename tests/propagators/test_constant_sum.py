import numpy as np

from ncs.problems.problem import ALGORITHM_CONSTANT_SUM, Problem, is_solved
from ncs.propagators.constant_sum_propagator import compute_domains


class TestConstant_sum:

    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10]], dtype=np.int32)
        data = np.array(8, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[8, 8]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array(8, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_alpha(self) -> None:
        problem = Problem(
            shared_domains=[
                2,
                18,
                13,
                20,
                17,
                1,
                25,
                21,
                22,
                6,
                19,
                4,
                9,
                11,
                10,
                24,
                8,
                12,
                15,
                5,
                3,
                23,
                7,
                26,
                14,
                16,
                19,
                37,
                52,
                3,
                18,
                8,
                7,
                50,
            ],
            domain_indices=list(range(34)),
            domain_offsets=[0] * 34,
        )
        problem.set_propagators([([0, 1, 4, 19, 31], ALGORITHM_CONSTANT_SUM, [45])])
        assert problem.filter()
        assert is_solved(problem.shared_domains)
