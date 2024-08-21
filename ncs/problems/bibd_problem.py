from ncs.problems.problem import Problem
from ncs.propagators.propagators import (
    ALG_EXACTLY_EQ,
    ALG_LEXICOGRAPHIC_LEQ,
    ALG_MIN_EQ,
)


class BIBDProblem(Problem):
    """
    A simple model for the BIBD problem.
    See https://www.csplib.org/Problems/prob028/ for a complete description of the problem.
    """

    def __init__(self, v: int, b: int, r: int, k: int, l: int):
        s = (v * (v - 1)) // 2
        n = (v + s) * b
        super().__init__(
            shr_domains=[(0, 1)] * n,
            dom_indices=list(range(n)),
            dom_offsets=[0] * n,
        )
        propagators = []
        # rows: counts
        for i in range(0, v):
            propagators.append((list(range(i * b, (i + 1) * b)), ALG_EXACTLY_EQ, [1, r]))
        # columns: counts
        for j in range(0, b):
            propagators.append((list(range(j, v * b, b)), ALG_EXACTLY_EQ, [1, k]))
        # scalar products: conjunctions and counts
        conj_idx = v * b
        for i1 in range(0, v - 1):
            for i2 in range(i1 + 1, v):
                conj_vars = []
                for j in range(0, b):
                    propagators.append(([(i1 * b + j), (i2 * b + j), conj_idx], ALG_MIN_EQ, []))
                    conj_vars.append(conj_idx)
                    conj_idx += 1
                propagators.append((conj_vars, ALG_EXACTLY_EQ, [1, l]))
        # remove symmetries
        # lexleq on rows
        for i in range(0, v - 1):
            propagators.append((list(range(i * b, (i + 2) * b)), ALG_LEXICOGRAPHIC_LEQ, []))
        # lexleq on columns
        for j in range(0, b - 1):
            propagators.append((list(range(j, v * b, b)) + list(range(j + 1, v * b, b)), ALG_LEXICOGRAPHIC_LEQ, []))
        self.set_propagators(propagators)
