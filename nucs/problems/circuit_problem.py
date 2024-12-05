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
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_NO_SUB_CYCLE


class CircuitProblem(Problem):
    """
    A model for circuit.
    """

    def __init__(self, n: int):
        """
        Inits the circuit problem.
        :param n: the number of vertices
        """
        self.n = n
        shr_domains = [(1, n - 1)] + [(0, n - 1)] * (n - 2) + [(0, n - 2)]
        super().__init__(shr_domains)
        s_indices = list(range(n))
        self.add_propagator((s_indices, ALG_ALLDIFFERENT, []))
        self.add_propagator((s_indices, ALG_NO_SUB_CYCLE, []))
        # self.add_propagator((s_indices, ALG_SCC, []))  # not worth the cost
