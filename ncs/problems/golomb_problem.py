from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import Problem
from ncs.propagators.propagators import ALG_AFFINE_EQ, ALG_AFFINE_LEQ, ALG_ALLDIFFERENT
from ncs.utils import MAX, MIN

GOLOMB_LENGTHS = [0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127]


def sum_first(n: int) -> int:
    return (n * (n + 1)) // 2


def index(mark_nb: int, i: int, j: int) -> int:
    return i * mark_nb - sum_first(i) + j - i - 1


def init_domains(var_nb: int, mark_nb: int) -> NDArray:
    domains = np.empty((var_nb, 2), dtype=np.int32)
    for i in range(0, mark_nb - 1):
        for j in range(i + 1, mark_nb):
            domains[index(mark_nb, i, j), MIN] = GOLOMB_LENGTHS[j - i + 1] if j - i + 1 < mark_nb else sum_first(j - i)
    domains[:, MAX] = 2**16
    return domains


class GolombProblem(Problem):
    """
    This is the famous Golomb ruler problem.
    It consists in finding n integers mark_i such that:
    - mark_0 = 0,
    - mark_0 <...< mark_n-1,
    - for all i<j, mark_j-mark_i are different,
    - mark_n-1 is minimal.
    Note: This is problem #6 in CSPLib.
    """

    def __init__(self, mark_nb: int) -> None:
        # dist_ij = mark_j - mark_i for j > i
        # mark_j = dist_0j for j > 0
        self.mark_nb = mark_nb
        self.length = mark_nb - 2
        var_nb = sum_first(mark_nb - 1)  # this the number of distances
        super().__init__(
            usr_shr_domains=init_domains(var_nb, mark_nb),
            usr_dom_indices=list(range(var_nb)),
            usr_dom_offsets=[0] * var_nb,
        )
        propagators: List[Tuple[List[int], int, List[int]]] = []
        # redundant constraints
        for i in range(mark_nb - 1):
            for j in range(i + 1, mark_nb):
                if j - i < mark_nb - 1:
                    propagators.append(
                        (
                            [index(mark_nb, i, j), self.length],
                            ALG_AFFINE_LEQ,
                            [-sum_first(mark_nb - 1 - (j - i)), 1, -1],
                        )
                    )
        # TODO break symmetries
        # main constraints
        for i in range(1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                propagators.append(
                    (
                        [index(mark_nb, 0, j), index(mark_nb, 0, i), index(mark_nb, i, j)],
                        ALG_AFFINE_EQ,
                        [0, 1, -1, -1],
                    )
                )
        propagators.append((list(range(var_nb)), ALG_ALLDIFFERENT, []))
        self.set_propagators(propagators)
