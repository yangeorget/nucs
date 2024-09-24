from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_EXACTLY_EQ, ALG_LEXICOGRAPHIC_LEQ, ALG_MIN_EQ


class BIBDProblem(Problem):
    """
    CSPLIB problem #28 - https://www.csplib.org/Problems/prob028/
    """

    def __init__(self, v: int, b: int, r: int, k: int, l: int):
        s = (v * (v - 1)) // 2
        n = (v + s) * b
        super().__init__([(0, 1)] * n)
        # rows: counts
        for i in range(0, v):
            self.add_propagator((list(range(i * b, (i + 1) * b)), ALG_EXACTLY_EQ, [1, r]))
        # columns: counts
        for j in range(0, b):
            self.add_propagator((list(range(j, v * b, b)), ALG_EXACTLY_EQ, [1, k]))
        # scalar products: conjunctions and counts
        conj_idx = v * b
        for i1 in range(0, v - 1):
            for i2 in range(i1 + 1, v):
                conj_vars = []
                for j in range(0, b):
                    self.add_propagator(
                        ([(i1 * b + j), (i2 * b + j), conj_idx], ALG_MIN_EQ, [])
                    )  # TODO:replace by AND ?
                    conj_vars.append(conj_idx)
                    conj_idx += 1
                self.add_propagator((conj_vars, ALG_EXACTLY_EQ, [1, l]))
        # remove symmetries
        # lexleq on rows
        for i in range(0, v - 1):
            self.add_propagator((list(range(i * b, (i + 2) * b)), ALG_LEXICOGRAPHIC_LEQ, []))
        # lexleq on columns
        for j in range(0, b - 1):
            self.add_propagator((list(range(j, v * b, b)) + list(range(j + 1, v * b, b)), ALG_LEXICOGRAPHIC_LEQ, []))
