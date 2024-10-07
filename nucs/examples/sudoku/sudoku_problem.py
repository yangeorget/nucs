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
from typing import List

from nucs.problems.latin_square_problem import LatinSquareProblem
from nucs.propagators.propagators import ALG_ALLDIFFERENT


class SudokuProblem(LatinSquareProblem):
    """
    A simple model for the sudoku problem.
    """

    def __init__(self, givens: List[List[int]]):
        """
        Inits the problem.
        :param givens: the givens per rows then per columns
        """
        super().__init__(list(range(1, 10)), givens)
        for i in range(3):
            for j in range(3):
                offset = i * 27 + j * 3
                self.add_propagator(
                    (
                        [
                            0 + offset,
                            1 + offset,
                            2 + offset,
                            9 + offset,
                            10 + offset,
                            11 + offset,
                            18 + offset,
                            19 + offset,
                            20 + offset,
                        ],
                        ALG_ALLDIFFERENT,
                        [],
                    )
                )
