from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_EXACTLY_EQ, ALG_LEXICOGRAPHIC_LEQ, ALG_MIN_EQ


class BIBDProblem(Problem):
    """
    CSPLIB problem #28 - https://www.csplib.org/Problems/prob028/
    """

    def __init__(self, object_nb: int, block_nb: int, by_row_count: int, by_column_count: int, scalar_product: int):
        self.object_nb = object_nb  # number of rows
        self.block_nb = block_nb  # number of columns
        matrix_var_nb = object_nb * block_nb  # number of cells in the matrix
        redundant_var_nb = ((object_nb * (object_nb - 1)) // 2) * block_nb
        super().__init__([(0, 1)] * (matrix_var_nb + redundant_var_nb))
        # rows: counts
        for object_idx in range(0, object_nb):
            self.add_propagator(
                (list(range(object_idx * block_nb, (object_idx + 1) * block_nb)), ALG_EXACTLY_EQ, [1, by_row_count])
            )
        # columns: counts
        for block_idx in range(0, block_nb):
            self.add_propagator(
                (list(range(block_idx, object_nb * block_nb, block_nb)), ALG_EXACTLY_EQ, [1, by_column_count])
            )
        # scalar products: conjunctions and counts
        conj_idx = object_nb * block_nb  # index of first redundant variable
        for i1 in range(0, object_nb - 1):
            for i2 in range(i1 + 1, object_nb):
                conj_vars = []
                for block_idx in range(0, block_nb):
                    self.add_propagator(
                        ([(i1 * block_nb + block_idx), (i2 * block_nb + block_idx), conj_idx], ALG_MIN_EQ, [])
                    )  # TODO:replace by AND ?
                    conj_vars.append(conj_idx)
                    conj_idx += 1
                self.add_propagator((conj_vars, ALG_EXACTLY_EQ, [1, scalar_product]))
        # remove symmetries
        # lexleq on rows
        for object_idx in range(0, object_nb - 1):
            self.add_propagator(
                (list(range(object_idx * block_nb, (object_idx + 2) * block_nb)), ALG_LEXICOGRAPHIC_LEQ, [])
            )
        # lexleq on columns
        for block_idx in range(0, block_nb - 1):
            self.add_propagator(
                (
                    list(range(block_idx, object_nb * block_nb, block_nb))
                    + list(range(block_idx + 1, object_nb * block_nb, block_nb)),
                    ALG_LEXICOGRAPHIC_LEQ,
                    [],
                )
            )

    def solution_as_matrix(self, solution: List[int]) -> List[List[int]]:
        return [[solution[i * self.block_nb + j] for j in range(0, self.block_nb)] for i in range(0, self.object_nb)]
