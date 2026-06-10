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
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AND_EQ, ALG_LEXLEQ, ALG_SUM_EQ_C


class BIBDProblem(Problem):
    """
    CSPLIB problem #28 - https://www.csplib.org/Problems/prob028/
    """

    def __init__(self, v: int, b: int, r: int, k: int, l: int, symmetry_breaking: bool = True):
        """
        Initializes the problem.

        :param v: the number of points/rows
        :type v: int
        :param b: the number of blocks
        :type b: int
        :param r: the number of true values per row
        :type r: int
        :param k: the number of true values per column
        :type k: int
        :param l: the scalar product between two rows
        :type l: int
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        :type symmetry_breaking: bool
        """
        self.v = v  # number of points/rows
        self.b = b  # number of blocks/columns
        matrix_var_nb = v * b  # number of cells in the matrix
        additional_var_nb = ((v * (v - 1)) >> 1) * b
        super().__init__([(0, 1)] * (matrix_var_nb + additional_var_nb))
        # rows: counts
        for object_idx in range(0, v):
            self.add_propagator(ALG_SUM_EQ_C, range(object_idx * b, (object_idx + 1) * b), [r])
        # columns: counts
        for block_idx in range(0, b):
            self.add_propagator(ALG_SUM_EQ_C, range(block_idx, v * b, b), [k])
        # scalar products: conjunctions and counts
        conj_idx = v * b  # index of first redundant variable
        for i1 in range(0, v - 1):
            for i2 in range(i1 + 1, v):
                conj_vars = []
                for block_idx in range(0, b):
                    self.add_propagator(ALG_AND_EQ, [i1 * b + block_idx, i2 * b + block_idx, conj_idx])
                    conj_vars.append(conj_idx)
                    conj_idx += 1
                self.add_propagator(ALG_SUM_EQ_C, conj_vars, [l])
        if symmetry_breaking:
            # lexleq on rows
            for object_idx in range(0, v - 1):
                self.add_propagator(ALG_LEXLEQ, range(object_idx * b, (object_idx + 2) * b))
            # lexleq on columns
            for block_idx in range(0, b - 1):
                self.add_propagator(ALG_LEXLEQ, list(range(block_idx, v * b, b)) + list(range(block_idx + 1, v * b, b)))

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns the solutions as a matrix of ints.

        :param solution: the solution as a list of ints
        :type solution: NDArray

        :return: a matrix
        :rtype: Any
        """
        solution_as_list = solution.tolist()
        return [[solution_as_list[i * self.b + j] for j in range(0, self.b)] for i in range(0, self.v)]
