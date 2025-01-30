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
from typing import Any, Dict

from numpy.typing import NDArray

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
        self.add_propagator(([A, B, D, E, G, L, N, O, R, T], ALG_ALLDIFFERENT, []))

    def solution_as_printable(self, solution: NDArray) -> Dict[str, Any]:
        """
        Returns the solution as a dict.
        :param solution: the solution as a list of ints
        :return: a dict
        """
        solution_as_list = solution.tolist()
        return {
            "A": solution_as_list[A],
            "B": solution_as_list[B],
            "D": solution_as_list[D],
            "E": solution_as_list[E],
            "G": solution_as_list[G],
            "L": solution_as_list[L],
            "N": solution_as_list[N],
            "O": solution_as_list[O],
            "R": solution_as_list[R],
            "T": solution_as_list[T],
        }
