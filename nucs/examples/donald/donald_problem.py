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
from typing import Dict

from numpy._typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT

A, B, D, E, G, L, N, O, R, T = tuple(range(10))


class DonaldProblem(Problem):
    """
    This is the famous crypto-arithmetic puzzle DONALD+GERALD=ROBERT.
    The letters represent digits that have to be all different.
    """

    def __init__(self) -> None:
        """
        Inits the problem.
        """
        super().__init__([(0, 9)] * 10)
        self.add_propagator(
            (
                [A, B, D, E, G, L, N, O, R, T],
                ALG_AFFINE_EQ,
                [200, -1000, 100002, 9900, 100000, 20, 1000, 0, -99010, -1, 0],
            )
        )
        self.add_propagator(
            ([A, B, D, E, G, L, N, O, R, T], ALG_ALLDIFFERENT, []),
        )

    def solution_as_dict(self, solution: NDArray) -> Dict[str, int]:
        """
        Returns the solution as a dict.
        :param solution: the solution as a list of ints
        :return: a dict
        """
        return {
            "A": solution[A],
            "B": solution[B],
            "D": solution[D],
            "E": solution[E],
            "G": solution[G],
            "L": solution[L],
            "N": solution[N],
            "O": solution[O],
            "R": solution[R],
            "T": solution[T],
        }
