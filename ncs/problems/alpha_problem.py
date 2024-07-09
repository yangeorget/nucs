from typing import List

from ncs.problems.problem import ALG_ALLDIFFERENT, ALG_CONSTANT_SUM, ALG_MUL, Problem

A = 0
B = 1
C = 2
D = 3
E = 4
F = 5
G = 6
H = 7
I = 8
J = 9
K = 10
L = 11
M = 12
N = 13
O = 14
P = 15
Q = 16
R = 17
S = 18
T = 19
U = 20
V = 21
W = 22
X = 23
Y = 24
Z = 25
CC = 26
EE = 27
II = 28
LL = 29
OO = 30
TT = 31
UU = 32
ZZ = 33


class AlphaProblem(Problem):
    """
    This problem comes from the newsgroup rec.puzzle.
    The numbers from 1 to 26 are assigned to the letters of the alphabet.
    The numbers beside each word are the total of the values assigned to the letters in the word
    (e.g for LYRE: L,Y,R,E might be to equal 5,9,20 and 13 or any other combination that add up to 47).

    Find the value of each letter under the equations:
    BALLET  45     GLEE  66     POLKA      59     SONG     61
    CELLO   43     JAZZ  58     QUARTET    50     SOPRANO  82
    CONCERT 74     LYRE  47     SAXOPHONE 134     THEME    72
    FLUTE   30     OBOE  53     SCALE      51     VIOLIN  100
    FUGUE   50     OPERA 65     SOLO       37     WALTZ    34
    """

    def __init__(self) -> None:
        super().__init__(
            shared_domains=[(1, 26)] * 26 + [(2, 52)] * 8,
            domain_indices=list(range(34)),
            domain_offsets=[0] * 34,
        )
        self.set_propagators(
            [
                ([A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z], ALG_ALLDIFFERENT, []),
                ([CC, C], ALG_MUL, [2]),
                ([EE, E], ALG_MUL, [2]),
                ([II, I], ALG_MUL, [2]),
                ([LL, L], ALG_MUL, [2]),
                ([OO, O], ALG_MUL, [2]),
                ([TT, T], ALG_MUL, [2]),
                ([UU, U], ALG_MUL, [2]),
                ([ZZ, Z], ALG_MUL, [2]),
                ([A, B, E, T, LL], ALG_CONSTANT_SUM, [45]),
                ([C, E, O, LL], ALG_CONSTANT_SUM, [43]),
                ([E, O, N, R, T, CC], ALG_CONSTANT_SUM, [74]),
                ([E, F, L, U, T], ALG_CONSTANT_SUM, [30]),
                ([E, F, G, UU], ALG_CONSTANT_SUM, [50]),
                ([G, L, EE], ALG_CONSTANT_SUM, [66]),
                ([A, J, ZZ], ALG_CONSTANT_SUM, [58]),
                ([E, L, R, Y], ALG_CONSTANT_SUM, [47]),
                ([E, B, OO], ALG_CONSTANT_SUM, [53]),
                ([A, E, P, O, R], ALG_CONSTANT_SUM, [65]),
                ([A, K, L, O, P], ALG_CONSTANT_SUM, [59]),
                ([A, E, Q, R, U, TT], ALG_CONSTANT_SUM, [50]),
                ([A, E, H, N, P, S, X, OO], ALG_CONSTANT_SUM, [134]),
                ([A, C, E, L, S], ALG_CONSTANT_SUM, [51]),
                ([L, S, OO], ALG_CONSTANT_SUM, [37]),
                ([G, N, O, S], ALG_CONSTANT_SUM, [61]),
                ([A, N, P, R, S, OO], ALG_CONSTANT_SUM, [82]),
                ([H, M, T, EE], ALG_CONSTANT_SUM, [72]),
                ([L, N, O, V, II], ALG_CONSTANT_SUM, [100]),
                ([A, L, T, W, Z], ALG_CONSTANT_SUM, [34]),
            ]
        )

    def pretty_print(self, solution: List[int]) -> None:
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
