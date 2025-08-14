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
from typing import Any, List

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_COUNT_EQ, ALG_COUNT_EQ_C, ALG_COUNT_LEQ_C


# TODO: GCC instead of booleans
class EmployeeSchedulingProblem(Problem):
    """
    See https://developers.google.com/optimization/scheduling/employee_scheduling.
    """

    def shift_index(self, day: int, shift: int, nurse: int) -> int:
        return day * self.shift_nb * self.nurse_nb + shift * self.nurse_nb + nurse

    def nurses(self, day: int, shift: int) -> List[int]:
        start_shift = self.shift_index(day, shift, 0)
        return list(range(start_shift, start_shift + self.nurse_nb))

    def shifts(self, day: int, nurse: int) -> List[int]:
        start_shift = self.shift_index(day, 0, nurse)
        return list(
            range(
                start_shift,
                start_shift + self.shift_nb * self.nurse_nb,
                self.nurse_nb,
            )
        )

    def __init__(self) -> None:
        """
        Initializes the problem.
        """
        self.day_nb = 7
        self.shift_nb = 3
        self.nurse_nb = 5
        self.shift_total_nb = self.day_nb * self.shift_nb * self.nurse_nb
        self.shift_requests_nds = [
            [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
        shift_requests_dsn = [
            self.shift_requests_nds[n][d][s]
            for d in range(self.day_nb)
            for s in range(self.shift_nb)
            for n in range(self.nurse_nb)
        ]
        self.requested_shifts = [iv[0] for iv in sorted(enumerate(shift_requests_dsn), key=lambda x: -x[1])]
        min_shift_count_per_nurse = (self.shift_nb * self.day_nb) // self.nurse_nb
        max_shift_count_per_nurse = min_shift_count_per_nurse + (
            0 if self.shift_nb * self.day_nb % self.nurse_nb == 0 else 1
        )
        super().__init__([(0, 1)] * self.shift_total_nb)  # the boolean variables for shifts
        self.add_variables(
            [(min_shift_count_per_nurse, max_shift_count_per_nurse)] * self.nurse_nb
        )  # the number of shifts per nurse
        self.satisfied_request_nb = self.add_variable((0, self.shift_total_nb))  # the number of satisfied requests
        self.add_propagators(
            [(self.nurses(d, s), ALG_COUNT_EQ_C, [1, 1]) for d in range(self.day_nb) for s in range(self.shift_nb)]
        )
        self.add_propagators(
            [(self.shifts(d, n), ALG_COUNT_LEQ_C, [1, 1]) for d in range(self.day_nb) for n in range(self.nurse_nb)]
        )
        self.add_propagators(
            [
                (list(range(n, self.shift_total_nb + self.nurse_nb, self.nurse_nb)), ALG_COUNT_EQ, [1])
                for n in range(self.nurse_nb)
            ]
        )
        self.add_propagator(
            (
                [index for index in range(0, self.shift_total_nb) if shift_requests_dsn[index]]
                + [self.satisfied_request_nb],
                ALG_COUNT_EQ,
                [1],
            )
        )

    def solution_as_printable(self, solution: NDArray) -> Any:
        return {
            d: [
                [
                    (("W" if self.shift_requests_nds[n][d][s] else "w") if solution[self.shift_index(d, s, n)] else " ")
                    for n in range(self.nurse_nb)
                ]
                for s in range(self.shift_nb)
            ]
            for d in range(self.day_nb)
        }
