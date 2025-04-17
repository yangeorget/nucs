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
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = tuple(range(26))


class AlphaProblem(Problem):
    """
    The numbers from 1 to 26 are assigned to the letters of the alphabet.
    The numbers beside each word are the total of the values assigned to the letters in the word
    (eg for LYRE: L,Y,R,E might be to equal 5,9,20 and 13 or any other combination that add up to 47).

    Find the value of each letter under the equations:
    BALLET  45     GLEE  66     POLKA      59     SONG     61
    CELLO   43     JAZZ  58     QUARTET    50     SOPRANO  82
    CONCERT 74     LYRE  47     SAXOPHONE 134     THEME    72
    FLUTE   30     OBOE  53     SCALE      51     VIOLIN  100
    FUGUE   50     OPERA 65     SOLO       37     WALTZ    34

    This problem comes from the newsgroup rec.puzzle.
    """

    def __init__(self) -> None:
        """
        Inits the problem.
        """
        super().__init__([(1, 26)] * 26)
        self.add_propagators(
            [
                ([A, B, E, T, L], ALG_AFFINE_EQ, [1, 1, 1, 1, 2, 45]),
                ([C, E, O, L], ALG_AFFINE_EQ, [1, 1, 1, 2, 43]),
                ([E, O, N, R, T, C], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 2, 74]),
                ([E, F, L, U, T], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 30]),
                ([E, F, G, U], ALG_AFFINE_EQ, [1, 1, 1, 2, 50]),
                ([G, L, E], ALG_AFFINE_EQ, [1, 1, 2, 66]),
                ([A, J, Z], ALG_AFFINE_EQ, [1, 1, 2, 58]),
                ([E, L, R, Y], ALG_AFFINE_EQ, [1, 1, 1, 1, 47]),
                ([E, B, O], ALG_AFFINE_EQ, [1, 1, 2, 53]),
                ([A, E, P, O, R], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 65]),
                ([A, K, L, O, P], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 59]),
                ([A, E, Q, R, U, T], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 2, 50]),
                ([A, E, H, N, P, S, X, O], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 1, 1, 2, 134]),
                ([A, C, E, L, S], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 51]),
                ([L, S, O], ALG_AFFINE_EQ, [1, 1, 2, 37]),
                ([G, N, O, S], ALG_AFFINE_EQ, [1, 1, 1, 1, 61]),
                ([A, N, P, R, S, O], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 2, 82]),
                ([H, M, T, E], ALG_AFFINE_EQ, [1, 1, 1, 2, 72]),
                ([L, N, O, V, I], ALG_AFFINE_EQ, [1, 1, 1, 1, 2, 100]),
                ([A, L, T, W, Z], ALG_AFFINE_EQ, [1, 1, 1, 1, 1, 34]),
                ([A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z], ALG_ALLDIFFERENT, []),
            ]
        )

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns the solution as a dict.
        :param solution: the solution as a list of ints
        :return: a dict
        """
        solution_as_list = solution.tolist()
        return {
            "A": solution_as_list[A],
            "B": solution_as_list[B],
            "C": solution_as_list[C],
            "D": solution_as_list[D],
            "E": solution_as_list[E],
            "F": solution_as_list[F],
            "G": solution_as_list[G],
            "H": solution_as_list[H],
            "I": solution_as_list[I],
            "J": solution_as_list[J],
            "K": solution_as_list[K],
            "L": solution_as_list[L],
            "M": solution_as_list[M],
            "N": solution_as_list[N],
            "O": solution_as_list[O],
            "P": solution_as_list[P],
            "Q": solution_as_list[Q],
            "R": solution_as_list[R],
            "S": solution_as_list[S],
            "T": solution_as_list[T],
            "U": solution_as_list[U],
            "V": solution_as_list[V],
            "W": solution_as_list[W],
            "X": solution_as_list[X],
            "Y": solution_as_list[Y],
            "Z": solution_as_list[Z],
        }
