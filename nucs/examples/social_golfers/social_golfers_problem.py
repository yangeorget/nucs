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


class SocialGolfersProblem(Problem):
    def index(self, w: int, p: int) -> int:
        return w * self.player_nb + p

    """
    CSPLIB problem #10 - https://www.csplib.org/Problems/prob010/
    """

    def __init__(self, group_nb: int, group_size: int, week_nb: int, symmetry_breaking: bool) -> None:
        """
        Initializes the problem.
        """
        self.group_nb = group_nb
        self.group_size = group_size
        self.week_nb = week_nb
        self.player_nb = group_nb * group_size
        domains = [(0, group_nb)] * self.player_nb * week_nb
        if symmetry_breaking:
            for p in range(self.player_nb):
                domains[self.index(0, p)] = p // group_size
            for k in range(group_size):
                for w in range(1, week_nb):
                    domains[self.index(w, k)] = k
        super().__init__(domains)
        for w1 in range(week_nb - 1):
            for w2 in range(w1 + 1, week_nb):
                for p1 in range(self.player_nb - 1):
                    for p2 in range(p1 + 1, self.player_nb):
                        # domains[index(w1, p1)] != domains[index(w1, p2)] or domains[index(w2, p1)] != domains[index(w2, p2)]
                        pass
        for wi in range(week_nb):
            # gcc(list(range(index(wi, 0), index(wi, playersNb), ...)
            pass
        if symmetry_breaking:
            # les domains sont lex increasing, pour chaque semaine
            pass
