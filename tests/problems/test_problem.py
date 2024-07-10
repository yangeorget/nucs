from ncs.problems.problem import ALG_ALLDIFFERENT, Problem, is_solved


class TestProblem:
    def test_is_not_solved_ok(self) -> None:
        problem = Problem(shared_domains=[(0, 2), (0, 2), (4, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        assert not is_solved(problem.shared_domains)

    def test_is_not_solved_ko(self) -> None:
        problem = Problem(shared_domains=[(2, 2), (2, 2), (6, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        assert is_solved(problem.shared_domains)

    def test_filter(self) -> None:
        problem = Problem(
            shared_domains=[(0, 0), (2, 2), (0, 2)],
            domain_indices=[0, 1, 2, 0, 1, 2],
            domain_offsets=[0, 0, 0, 0, 1, 2],
        )
        problem.set_propagators(
            [
                ([0, 1, 2], ALG_ALLDIFFERENT, []),
                ([3, 4, 5], ALG_ALLDIFFERENT, []),
            ]
        )
        assert not problem.filter()
