import numpy as np

from ncs.problems.problem import (
    ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ,
    ALGORITHM_SUM,
    Problem,
    is_solved,
)


class TestProblem:
    def test_is_not_solved_ok(self) -> None:
        problem = Problem(shared_domains=[(0, 2), (0, 2), (4, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        assert not is_solved(problem.shared_domains)

    def test_is_not_solved_ko(self) -> None:
        problem = Problem(shared_domains=[(2, 2), (2, 2), (6, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        assert is_solved(problem.shared_domains)

    def test_filter_1(self) -> None:
        problem = Problem(shared_domains=[(0, 2), (0, 2), (4, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        problem.set_propagators([([2, 0, 1], ALGORITHM_SUM, [])])
        assert problem.filter()
        assert is_solved(problem.shared_domains)
        assert np.all(problem.shared_domains == np.array([[2, 2], [2, 2], [4, 4]]))

    def test_filter_2(self) -> None:
        problem = Problem(
            shared_domains=[(0, 0), (2, 2), (0, 2)],
            domain_indices=[0, 1, 2, 0, 1, 2],
            domain_offsets=[0, 0, 0, 0, 1, 2],
        )
        problem.set_propagators(
            [
                ([0, 1, 2], ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ, []),
                ([3, 4, 5], ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ, []),
            ]
        )
        assert not problem.filter()
