from pprint import pprint
from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = tuple(range(26))


class AlphaProblem(Problem):
    """
    The numbers from 1 to 26 are assigned to the letters of the alphabet.
    The numbers beside each word are the total of the values assigned to the letters in the word
    (e.g for LYRE: L,Y,R,E might be to equal 5,9,20 and 13 or any other combination that add up to 47).

    Find the value of each letter under the equations:
    BALLET  45     GLEE  66     POLKA      59     SONG     61
    CELLO   43     JAZZ  58     QUARTET    50     SOPRANO  82
    CONCERT 74     LYRE  47     SAXOPHONE 134     THEME    72
    FLUTE   30     OBOE  53     SCALE      51     VIOLIN  100
    FUGUE   50     OPERA 65     SOLO       37     WALTZ    34

    This problem comes from the newsgroup rec.puzzle.
    """

    def __init__(self) -> None:
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

    def pretty_print_solution(self, solution: List[int]) -> None:
        print(
            {
                "A": solution[A],
                "B": solution[B],
                "C": solution[C],
                "D": solution[D],
                "E": solution[E],
                "F": solution[F],
                "G": solution[G],
                "H": solution[H],
                "I": solution[I],
                "J": solution[J],
                "K": solution[K],
                "L": solution[L],
                "M": solution[M],
                "N": solution[N],
                "O": solution[O],
                "P": solution[P],
                "Q": solution[Q],
                "R": solution[R],
                "S": solution[S],
                "T": solution[T],
                "U": solution[U],
                "V": solution[V],
                "W": solution[W],
                "X": solution[X],
                "Y": solution[Y],
                "Z": solution[Z],
            }
        )


if __name__ == "__main__":
    problem = AlphaProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solutions = solver.find_all()
    pprint(get_statistics(solver.statistics))
    print(solutions[0])