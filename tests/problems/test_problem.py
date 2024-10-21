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

    def test_split_0(self) -> None:
        problem = Problem([(0, 9)])
        problems = problem.split(2, 0)
        assert problems[0].shr_domains_lst[0] == (0, 4)
        assert problems[1].shr_domains_lst[0] == (5, 9)

    def test_split_1(self) -> None:
        problem = Problem([(0, 9)])
        problems = problem.split(3, 0)
        assert problems[0].shr_domains_lst[0] == (0, 3)
        assert problems[1].shr_domains_lst[0] == (4, 6)
        assert problems[2].shr_domains_lst[0] == (7, 9)

    def test_split_2(self) -> None:
        problem = Problem([(0, 11)])
        problems = problem.split(8, 0)
        assert problems[0].shr_domains_lst[0] == (0, 1)
        assert problems[1].shr_domains_lst[0] == (2, 3)
        assert problems[2].shr_domains_lst[0] == (4, 5)
        assert problems[3].shr_domains_lst[0] == (6, 7)
        assert problems[4].shr_domains_lst[0] == (8, 8)
        assert problems[5].shr_domains_lst[0] == (9, 9)
        assert problems[6].shr_domains_lst[0] == (10, 10)
        assert problems[7].shr_domains_lst[0] == (11, 11)
