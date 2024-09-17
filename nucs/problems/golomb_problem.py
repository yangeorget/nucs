import numpy as np
from numpy.typing import NDArray

from nucs.constants import MAX, MIN
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_AFFINE_LEQ, ALG_ALLDIFFERENT
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic

GOLOMB_LENGTHS = [0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127]


def sum_first(n: int) -> int:
    return (n * (n + 1)) // 2


def index(mark_nb: int, i: int, j: int) -> int:
    return i * mark_nb - sum_first(i) + j - i - 1


def init_domains(dist_nb: int, mark_nb: int) -> NDArray:
    domains = np.empty((dist_nb, 2), dtype=np.int32, order="F")
    for i in range(0, mark_nb - 1):
        for j in range(i + 1, mark_nb):
            domains[index(mark_nb, i, j), MIN] = GOLOMB_LENGTHS[j - i + 1] if j - i + 1 < mark_nb else sum_first(j - i)
    domains[:, MAX] = sum_first(dist_nb)
    return domains


class GolombProblem(Problem):
    """
    This is the famous Golomb ruler problem.

    It consists in finding n integers mark_i such that:
    - mark_0 = 0,
    - mark_0 <...< mark_n-1,
    - for all i < j, mark_j - mark_i are different,
    - mark_n-1 is minimal.

    CSPLIB problem #6 - https://www.csplib.org/Problems/prob006/
    """

    def __init__(self, mark_nb: int) -> None:
        self.mark_nb = mark_nb
        self.dist_nb = sum_first(mark_nb - 1)  # this the number of distances
        super().__init__(init_domains(self.dist_nb, mark_nb))
        self.length_idx = index(mark_nb, 0, mark_nb - 1)  # we want to minimize this
        # Additional items used in prune():
        # - a reusable array for storing the minimal sum of different integers:
        # minimal_sum[i] will be the minimal sum of i different integers chosen among a set of possible integers
        self.minimal_sum = np.zeros(self.mark_nb - 1, dtype=int)
        # - a boolean array for marking already existing distances as used
        self.used_distance = np.empty(sum_first(self.mark_nb - 2) + 1, dtype=bool)
        # main constraints
        # dist_ij = mark_j - mark_i for j > i
        # mark_j = dist_0j for j > 0
        for i in range(1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                self.add_propagator(
                    (
                        [index(mark_nb, 0, j), index(mark_nb, 0, i), index(mark_nb, i, j)],
                        ALG_AFFINE_EQ,
                        [1, -1, -1, 0],
                    )
                )
        self.add_propagator((list(range(self.dist_nb)), ALG_ALLDIFFERENT, []))
        # redundant constraints
        for i in range(mark_nb - 1):
            for j in range(i + 1, mark_nb):
                if j - i < mark_nb - 1:
                    self.add_propagator(
                        (
                            [index(mark_nb, i, j), index(mark_nb, 0, mark_nb - 1)],
                            ALG_AFFINE_LEQ,
                            [1, -1, -sum_first(mark_nb - 1 - (j - i))],
                        ),
                        0,
                    )
        # break symmetries
        self.add_propagator(
            (
                [index(mark_nb, 0, 1), index(mark_nb, mark_nb - 2, mark_nb - 1)],
                ALG_AFFINE_LEQ,
                [1, -1, -1],
            ),
            0,
        )

    def prune(self) -> bool:
        """
        A method for pruning the search space of the Golomb problem.
        """
        ni_var_idx = first_not_instantiated_var_heuristic(self.shr_domains_arr, self.dom_indices_arr)
        if 1 < ni_var_idx < self.mark_nb - 1:  # otherwise useless
            self.used_distance.fill(False)
            # the following will mark at most sum(n-3) numbers as used
            # hence there will be at least n-2 unused numbers greater than 0
            for var_idx in range(index(self.mark_nb, ni_var_idx - 2, ni_var_idx - 1) + 1):
                dist = self.get_min_value(var_idx)
                if dist < len(self.used_distance):
                    self.used_distance[dist] = True
            # let's compute the sum of non-used numbers
            distance = 1
            for j in range(0, self.mark_nb - ni_var_idx):
                while self.used_distance[distance]:
                    distance += 1
                self.minimal_sum[j + 1] = self.minimal_sum[j] + distance
                distance += 1
            for i in range(ni_var_idx - 1, self.mark_nb - 1):
                for j in range(i + 1, self.mark_nb):
                    var_idx = index(self.mark_nb, i, j)
                    new_min = self.minimal_sum[j - i]
                    # if new_min > self.get_max_value(var_idx):  # a bit slower
                    #    return False
                    self.set_min_value(var_idx, new_min)
        return True
