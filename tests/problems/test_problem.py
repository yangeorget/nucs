from nucs.constants import MAX, MIN
from nucs.numpy import new_shr_domain_changes
from nucs.problems.problem import Problem, is_solved
from nucs.propagators.propagators import ALG_AFFINE_EQ


class TestProblem:
    def test_set_min_value(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)], [0, 1, 2], [1, 1, 1])
        problem.init_problem()
        problem.set_min_value(1, 2)
        assert problem.shr_domains_arr[1, MIN] == 1
        assert problem.shr_domains_arr[1, MAX] == 2

    def test_is_not_solved_ok(self) -> None:
        problem = Problem([(0, 2), (0, 2), (4, 6)])
        problem.init_problem()
        assert not is_solved(problem.shr_domains_arr)

    def test_is_not_solved_ko(self) -> None:
        problem = Problem([(2, 2), (2, 2), (6, 6)])
        problem.init_problem()
        assert is_solved(problem.shr_domains_arr)

    def test_filter(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_AFFINE_EQ, [3, 1, 2, 5]))
        problem.add_propagator(([0, 1, 2], ALG_AFFINE_EQ, [2, 1, 2, 4]))
        problem.add_propagator(([1, 2], ALG_AFFINE_EQ, [1, 1, 1]))
        shr_domain_changes = new_shr_domain_changes(3)
        problem.filter(shr_domain_changes)
        assert is_solved(problem.shr_domains_arr)
        assert problem.shr_domains_arr[0][MIN] == 1
        assert problem.shr_domains_arr[0][MAX] == 1
        assert problem.shr_domains_arr[1][MIN] == 0
        assert problem.shr_domains_arr[1][MAX] == 0
        assert problem.shr_domains_arr[2][MIN] == 1
        assert problem.shr_domains_arr[2][MAX] == 1
