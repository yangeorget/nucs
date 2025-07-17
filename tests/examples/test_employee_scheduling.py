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
        assert EmployeeSchedulingProblem().nurses(0, 0) == [0, 1, 2, 3]
        assert EmployeeSchedulingProblem().nurses(0, 1) == [4, 5, 6, 7]
        assert EmployeeSchedulingProblem().nurses(1, 0) == [12, 13, 14, 15]
        assert EmployeeSchedulingProblem().nurses(1, 1) == [16, 17, 18, 19]

    def test_shifts(self) -> None:
        assert EmployeeSchedulingProblem().shifts(0, 0) == [0, 4, 8]
        assert EmployeeSchedulingProblem().shifts(0, 1) == [1, 5, 9]
        assert EmployeeSchedulingProblem().shifts(1, 0) == [12, 16, 20]
        assert EmployeeSchedulingProblem().shifts(1, 1) == [13, 17, 21]
