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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from nucs.examples.employee_scheduling.employee_scheduling_problem import EmployeeSchedulingProblem


class TestEmployeeScheduling:
    def test_nurses(self) -> None:
        assert EmployeeSchedulingProblem().nurses(0, 0) == [0, 1, 2, 3, 4]
        assert EmployeeSchedulingProblem().nurses(0, 1) == [5, 6, 7, 8, 9]
        assert EmployeeSchedulingProblem().nurses(1, 0) == [15, 16, 17, 18, 19]
        assert EmployeeSchedulingProblem().nurses(1, 1) == [20, 21, 22, 23, 24]

    def test_shifts(self) -> None:
        assert EmployeeSchedulingProblem().shifts(0, 0) == [0, 5, 10]
        assert EmployeeSchedulingProblem().shifts(0, 1) == [1, 6, 11]
        assert EmployeeSchedulingProblem().shifts(1, 0) == [15, 20, 25]
        assert EmployeeSchedulingProblem().shifts(1, 1) == [16, 21, 26]
