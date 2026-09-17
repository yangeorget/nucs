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
from nucs.problems.permutation_problem import PermutationProblem
from nucs.propagators.propagators import ALG_NO_SUB_CYCLE


class CircuitProblem(PermutationProblem):
    """
    A model for circuits: the successors and the predecessors are each a permutation forming a single circuit,
    enforced by NO_SUB_CYCLE.

    CIRCUIT_POSITIONS, which the FlatZinc circuit uses, also bounds the position of each node on the tour. That pays
    when other constraints leave each node few possible successors (time windows, scheduling); on a complete graph,
    as in the TSP examples, every node is one step from every other, no position is ever ruled out, and it only
    costs time -- about 40% on gr17, gr21 and gr24.
    """

    def __init__(self, n: int):
        """
        Initializes the circuit problem.

        :param n: the number of vertices
        :type n: int
        """
        self.n = n
        super().__init__(n)
        self.domains[0] = (1, n - 1)
        self.domains[n - 1] = (0, n - 2)
        self.domains[n] = (1, n - 1)
        self.domains[2 * n - 1] = (0, n - 2)
        self.add_propagator(ALG_NO_SUB_CYCLE, range(n))
        self.add_propagator(ALG_NO_SUB_CYCLE, range(n, 2 * n))
        # self.add_propagator((list(range(n)), ALG_SCC, []))  # not worth the cost
        # self.add_propagator((list(range(n, 2*n)), ALG_SCC, []))  # not worth the cost
