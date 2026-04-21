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

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_EQUIV_EQ, ALG_SUM_LEQ_C, ALG_GCC, ALG_LEXICOGRAPHIC_LEQ


class SocialGolfersProblem(Problem):
    """
    CSPLIB problem #10 - https://www.csplib.org/Problems/prob010/
    """

    def index(self, w: int, p: int) -> int:
        return w * self.player_nb + p

    def bool_index(self, w: int, p1: int, p2: int) -> int:
        return (
            w * self.player_nb * (self.player_nb - 1) // 2
            + self.player_nb * p1
            - p1 * (p1 + 1) // 2
            + p2
            - p1
            - 1
            + self.player_nb * self.week_nb
        )

    def __init__(self, group_nb: int, group_size: int, week_nb: int, symmetry_breaking: bool) -> None:
        """
        Initializes the problem.
        """
        self.group_nb = group_nb
        self.group_size = group_size
        self.week_nb = week_nb
        self.player_nb = group_nb * group_size
        domains = [(0, group_nb - 1)] * self.player_nb * week_nb + [(0, 1)] * (
            week_nb * self.player_nb * (self.player_nb - 1) // 2
        )
        if symmetry_breaking:
            for p in range(self.player_nb):
                domains[self.index(0, p)] = (p // group_size, p // group_size)
            for k in range(group_size):
                for w in range(1, week_nb):
                    domains[self.index(w, k)] = (k, k)
        super().__init__(domains)
        for w1 in range(week_nb - 1):
            for w2 in range(w1 + 1, week_nb):
                for p1 in range(self.player_nb - 1):
                    for p2 in range(p1 + 1, self.player_nb):
                        self.add_propagator(
                            ALG_EQUIV_EQ, [self.bool_index(w1, p1, p2), self.index(w1, p1), self.index(w1, p2)]
                        )
                        self.add_propagator(
                            ALG_EQUIV_EQ, [self.bool_index(w2, p1, p2), self.index(w2, p1), self.index(w2, p2)]
                        )
                        self.add_propagator(
                            ALG_SUM_LEQ_C, [self.bool_index(w1, p1, p2), self.bool_index(w2, p1, p2)], [1]
                        )

        for w in range(week_nb):
            self.add_propagator(
                ALG_GCC,
                list(range(self.index(w, 0), self.index(w, self.player_nb))),
                [0, *([self.group_size] * 2 * self.group_nb)],
            )
        if symmetry_breaking:
            for w in range(week_nb - 1):
                self.add_propagator(ALG_LEXICOGRAPHIC_LEQ, [*range(self.player_nb * w, self.player_nb * (w + 2))])
