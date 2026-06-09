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
from nucs.propagators.propagators import ALG_LINEAR_EQ_C, ALG_LINEAR_LEQ_C


class KnapsackProblem(Problem):
    """
    CSPLIB problem #133 - https://www.csplib.org/Problems/prob133/
    """

    def __init__(self, dataset: dict) -> None:
        """
        Inits the problem.

        :param dataset: the dataset
        :type dataset: dict
        """
        capacity = dataset["capacity"]
        weights = dataset["weights"]
        volumes = dataset["volumes"]
        n = len(weights)
        super().__init__([(0, 1)] * n + [(0, sum(weights))])
        self.add_propagator(ALG_LINEAR_LEQ_C, range(n), [*volumes, capacity])
        self.add_propagator(ALG_LINEAR_EQ_C, range(n + 1), [*weights, -1, 0])
        self.weight = n
