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

from numpy._typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_EXACTLY_EQ, ALG_GCC, ALG_RELATION


class SportsTournamentSchedulingProblem(Problem):
    """
    The problem is to schedule a tournament of n teams over n−1 weeks,
    with each week divided into n/2 periods,
    and each period divided into two slots.

    The first team in each slot plays at home, whilst the second plays the first team away.

    A tournament must satisfy the following three constraints:
    - every team plays once a week;
    - every team plays at most twice in the same period over the tournament;
    - every team plays every other team.

    CSPLIB problem #26 - https://www.csplib.org/Problems/prob026/
    """

    def solution_as_matrix(self, solution: NDArray) -> List[List[str]]:
        """
        Returns the solutions as a matrix of strings.
        :param solution: the solution as a list of ints
        :return: a matrix
        """
        return [
            [
                f"{solution[self.team_var_index(p, w, 0)]}-{solution[self.team_var_index(p, w, 1)]}"
                for w in range(0, self.week_nb)
            ]
            for p in range(0, self.period_nb)
        ]

    def team_var_index(self, p: int, w: int, s: int) -> int:
        return p * (self.week_nb * self.slot_nb) + w * self.slot_nb + s

    def match_var_index(self, p: int, w: int) -> int:
        return self.team_var_nb + p * self.week_nb + w

    def teams_per_week(self, w: int) -> List[int]:
        return [self.team_var_index(p, w, s) for p in range(self.period_nb) for s in range(self.slot_nb)]

    def teams_per_period(self, p: int) -> List[int]:
        return [self.team_var_index(p, w, s) for w in range(self.week_nb) for s in range(self.slot_nb)]

    def matches(self) -> List[int]:
        return list(range(self.team_var_nb, self.team_var_nb + self.match_nb))

    def matches_per_week(self, w: int) -> List[int]:
        return [self.match_var_index(p, w) for p in range(self.period_nb)]

    def match_ordinal(self, t1: int, t2: int) -> int:
        return self.match_nb - ((self.team_nb - t1) * (self.team_nb - t1 - 1)) // 2 + t2 - t1 - 1

    def plays(self) -> List[int]:
        plays = []
        for i in range(0, self.team_nb - 1):
            for j in range(i + 1, self.team_nb):
                plays.extend([i, j, self.match_ordinal(i, j)])
        return plays

    def __init__(self, n: int, symmetry_breaking: bool = True) -> None:
        self.team_nb = n
        self.slot_nb = 2
        self.period_nb = n // 2
        self.week_nb = n - 1
        self.match_nb = ((n - 1) * n) // 2
        self.team_var_nb = self.period_nb * self.week_nb * self.slot_nb
        super().__init__([(0, self.team_nb - 1)] * self.team_var_nb + [(0, self.match_nb - 1)] * self.match_nb)
        plays = self.plays()
        self.add_propagator((self.matches(), ALG_ALLDIFFERENT, []))  # matches are different
        self.add_propagators([(self.teams_per_week(w), ALG_ALLDIFFERENT, []) for w in range(0, self.week_nb)])
        self.add_propagators(
            [(self.teams_per_period(p), ALG_GCC, [0] + [1] * n + [2] * n) for p in range(0, self.period_nb)]
        )
        self.add_propagators(
            [
                (
                    [self.team_var_index(p, w, 0), self.team_var_index(p, w, 1), self.match_var_index(p, w)],
                    ALG_RELATION,
                    plays,
                )
                for p in range(0, self.period_nb)
                for w in range(0, self.week_nb)
            ]
        )
        if symmetry_breaking:
            # the first week is set
            k = 0
            for p in range(self.period_nb):
                for s in range(self.slot_nb):
                    self.shr_domains_lst[self.team_var_index(p, 0, s)] = [k, k]
                    k += 1
            # the match `0 versus (t+1)` appears at week t (much slower)
            self.add_propagators(
                [
                    (self.matches_per_week(w), ALG_EXACTLY_EQ, [self.match_ordinal(0, w + 1), 1])
                    for w in range(self.week_nb)
                ]
            )
