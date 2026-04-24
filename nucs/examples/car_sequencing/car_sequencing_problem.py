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
# Copyright 2024-2026 - Yan Georget
###############################################################################
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ELEMENT_EQ, ALG_GCC, ALG_SUM_LEQ_C


class CarSequencingProblem(Problem):
    """
    CSPLIB problem #1 - https://www.csplib.org/Problems/prob001/

    Cars must be sequenced on an assembly line. For each option o, in any
    consecutive block_size[o] cars, at most max_per_block[o] may require option o.
    The number of cars of each class must match the given demands.

    Variables:
    - slot[i] in [0, class_nb-1]: the class of the car at position i
    - option[i][o] in {0, 1}: whether the car at position i requires option o

    Constraints:
    - GCC on slots enforces class demands
    - ELEMENT_EQ links option[i][o] to slot[i] via the requires table
    - SUM_LEQ_C enforces sliding-window capacity limits per option
    """

    def slot_var(self, i: int) -> int:
        return i

    def option_var(self, i: int, o: int) -> int:
        return self.car_nb + i * self.option_nb + o

    def __init__(self, dataset: dict) -> None:
        self.car_nb = dataset["car_nb"]
        self.option_nb = dataset["option_nb"]
        self.class_nb = dataset["class_nb"]
        domains = [(0, self.class_nb - 1)] * self.car_nb + [(0, 1)] * (self.car_nb * self.option_nb)
        super().__init__(domains)
        # Demand: exactly demands[c] cars of each class c
        gcc_params = [0]
        for d in dataset["demands"]:
            gcc_params += [d, d]
        self.add_propagator(ALG_GCC, list(range(self.car_nb)), gcc_params)
        # Linking: option[i][o] = requires[slot[i]][o]
        for o in range(self.option_nb):
            option_col = [dataset["requires"][c][o] for c in range(self.class_nb)]
            for i in range(self.car_nb):
                self.add_propagator(
                    ALG_ELEMENT_EQ,
                    [self.slot_var(i), self.option_var(i, o)],
                    option_col,
                )
        # Capacity: sum of option[j..j+block_size[o]-1][o] <= max_per_block[o]
        for o in range(self.option_nb):
            b = dataset["block_size"][o]
            m = dataset["max_per_block"][o]
            for j in range(self.car_nb - b + 1):
                self.add_propagator(
                    ALG_SUM_LEQ_C,
                    [self.option_var(j + k, o) for k in range(b)],
                    [m],
                )

    def solution_as_printable(self, solution: NDArray) -> Any:
        return [int(solution[self.slot_var(i)]) for i in range(self.car_nb)]
