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


class EmployeeSchedulingProblem(Problem):
    """ """

    def nurses(self, day: int, shift: int) -> List[int]:
        return list(
            range(
                day * self.shift_nb * self.nurse_nb + shift * self.nurse_nb,
                day * self.shift_nb * self.nurse_nb + (shift + 1) * self.nurse_nb,
            )
        )

    def shifts(self, day: int, nurse: int) -> List[int]:
        return list(
            range(
                day * self.shift_nb * self.nurse_nb + nurse,
                (day + 1) * self.shift_nb * self.nurse_nb + nurse,
                self.nurse_nb,
            )
        )

    def __init__(self) -> None:
        """
        Initializes the problem.
        """
        self.day_nb = 3
        self.shift_nb = 3
        self.nurse_nb = 4
        min_shifts_per_nurse = (self.shift_nb * self.day_nb) // self.nurse_nb
        if self.shift_nb * self.day_nb % self.nurse_nb == 0:
            max_shifts_per_nurse = min_shifts_per_nurse
        else:
            max_shifts_per_nurse = min_shifts_per_nurse + 1
        super().__init__(
            [(0, 1)] * self.nurse_nb * self.shift_nb * self.day_nb
            + [(min_shifts_per_nurse, max_shifts_per_nurse)] * self.nurse_nb
        )
        for d in range(self.day_nb):
            for s in range(self.shift_nb):
                self.add_propagator((self.nurses(d, s), ALG_COUNT_EQ_C, [1]))
        for d in range(self.day_nb):
            for n in range(self.nurse_nb):
                self.add_propagator((self.shifts(d, n), ALG_COUNT_LEQ_C, [1]))
        for n in range(self.nurse_nb):
            self.add_propagator(
                (list(range(n, self.nurse_nb * (self.shift_nb * self.day_nb + 1), self.nurse_nb)), ALG_COUNT_EQ, [1])
            )

    def solution_as_printable(self, solution: NDArray) -> Any:
        return {
            d: [
                [
                    "x" if solution[d * self.shift_nb * self.nurse_nb + s * self.nurse_nb + n] else " "
                    for n in range(self.nurse_nb)
                ]
                for s in range(self.shift_nb)
            ]
            for d in range(self.day_nb)
        }
