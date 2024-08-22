from nucs.memory import MAX, MIN
from nucs.problems.problem import Problem, is_solved


class TestProblem:
    def test_set_min_value(self) -> None:
        problem = Problem(shr_domains=[(0, 2), (0, 2), (0, 2)], dom_indices=[0, 1, 2], dom_offsets=[1, 1, 1])
        problem.set_min_value(1, 2)
        assert problem.shr_domains[1, MIN] == 1
        assert problem.shr_domains[1, MAX] == 2

    def test_is_not_solved_ok(self) -> None:
        problem = Problem(shr_domains=[(0, 2), (0, 2), (4, 6)], dom_indices=[0, 1, 2], dom_offsets=[0, 0, 0])
        assert not is_solved(problem.shr_domains)

    def test_is_not_solved_ko(self) -> None:
        problem = Problem(shr_domains=[(2, 2), (2, 2), (6, 6)], dom_indices=[0, 1, 2], dom_offsets=[0, 0, 0])
        assert is_solved(problem.shr_domains)
