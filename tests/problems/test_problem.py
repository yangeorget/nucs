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
from nucs.constants import MAX, MIN
from nucs.problems.problem import Problem, is_solved


class TestProblem:
    def test_set_min_value(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)], [0, 1, 2], [1, 1, 1])
        problem.init()
        problem.set_min_value(1, 2)
        assert problem.shr_domains_arr[1, MIN] == 1
        assert problem.shr_domains_arr[1, MAX] == 2

    def test_is_not_solved_ok(self) -> None:
        problem = Problem([(0, 2), (0, 2), (4, 6)])
        problem.init()
        assert not is_solved(problem.shr_domains_arr)

    def test_is_not_solved_ko(self) -> None:
        problem = Problem([(2, 2), (2, 2), (6, 6)])
        problem.init()
        assert is_solved(problem.shr_domains_arr)
