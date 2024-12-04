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
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_SCC


class CircuitProblem(Problem):
    """
    A model for circuits:
    - there are n vertices
    - s_i is the successor of node i for i in [0, n-1]
    - t corresponds the successors of vertex 0
    - t_0 = s_0, t_{i+1} is the successor of t_i for in [0, n-2]
    - thus we need n-2 additional variables only
    """

    def __init__(self, n: int):
        """
        Inits the circuit problem.
        :param n: the number of vertices
        """
        self.n = n
        shr_domains = [(1, n - 1)] + [(0, n - 1)] * (n - 2) + [(0, n - 2)]  # + [(1, n - 1)] * (n - 2)
        super().__init__(shr_domains)
        s_indices = list(range(n))
        self.add_propagator((s_indices, ALG_ALLDIFFERENT, []))
        # t_indices = [0] + list(range(n, 2 * n - 2))
        # self.add_propagator((t_indices, ALG_ALLDIFFERENT, []))
        # for i in range(n - 3):
        #     self.add_propagator((s_indices + [t_indices[i], t_indices[i + 1]], ALG_ELEMENT_LIV, []))
        # self.add_propagator((s_indices + [t_indices[-1]], ALG_ELEMENT_LIC, [0]))
        self.add_propagator((s_indices, ALG_SCC, []))
        # TODO: add a nocycle constraint ?
