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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AND_EQ, ALG_COUNT_EQ_C, ALG_LEXICOGRAPHIC_LEQ


class BIBDProblem(Problem):
    """
    CSPLIB problem #28 - https://www.csplib.org/Problems/prob028/
    """

    def __init__(self, v: int, b: int, r: int, k: int, l: int, symmetry_breaking: bool = True):
        """
        Initializes the problem.
        :param v: the number of points/rows
        :param b: the number of blocks
        :param r: the number of true values per row
        :param k: the number of true values per column
        :param l: the scalar product between two rows
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        self.v = v  # number of points/rows
        self.b = b  # number of blocks/columns
        matrix_var_nb = v * b  # number of cells in the matrix
        additional_var_nb = ((v * (v - 1)) // 2) * b
        super().__init__([(0, 1)] * (matrix_var_nb + additional_var_nb))
        # rows: counts
        for object_idx in range(0, v):
            self.add_propagator((list(range(object_idx * b, (object_idx + 1) * b)), ALG_COUNT_EQ_C, [1, r]))
        # columns: counts
        for block_idx in range(0, b):
            self.add_propagator((list(range(block_idx, v * b, b)), ALG_COUNT_EQ_C, [1, k]))
        # scalar products: conjunctions and counts
        conj_idx = v * b  # index of first redundant variable
        for i1 in range(0, v - 1):
            for i2 in range(i1 + 1, v):
                conj_vars = []
                for block_idx in range(0, b):
                    self.add_propagator(([i1 * b + block_idx, i2 * b + block_idx, conj_idx], ALG_AND_EQ, []))
                    conj_vars.append(conj_idx)
                    conj_idx += 1
                self.add_propagator((conj_vars, ALG_COUNT_EQ_C, [1, l]))
        if symmetry_breaking:
            # lexleq on rows
            for object_idx in range(0, v - 1):
                self.add_propagator((list(range(object_idx * b, (object_idx + 2) * b)), ALG_LEXICOGRAPHIC_LEQ, []))
            # lexleq on columns
            for block_idx in range(0, b - 1):
                self.add_propagator(
                    (
                        list(range(block_idx, v * b, b)) + list(range(block_idx + 1, v * b, b)),
                        ALG_LEXICOGRAPHIC_LEQ,
                        [],
                    )
                )

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns the solutions as a matrix of ints.
        :param solution: the solution as a list of ints
        :return: a matrix
        """
        solution_as_list = solution.tolist()
        return [[solution_as_list[i * self.b + j] for j in range(0, self.b)] for i in range(0, self.v)]
