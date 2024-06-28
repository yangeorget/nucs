from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.propagators import (
    alldifferent_lopez_ortiz_propagator,
    dummy_propagator,
    sum_propagator,
)

ALLDIFFERENT_LOPEZ_ORTIZ = 0
DUMMY = 1
SUM = 2


# @jitclass(
#     [
#         ("size", numba.int32),
#         ("variables", numba.int32[:]),
#         ("triggers", numba.boolean[:, :]),
#         ("offsets", numba.int32[:]),
#         ("indices", numba.int32[:]),
#         ("algorithm", numba.int32),
#     ]
# )
class Propagator:
    def __init__(self, variables: NDArray, algorithm: int):
        self.size = len(variables)
        self.variables = variables
        self.triggers = np.ones((self.size, 2), dtype=np.bool)
        self.offsets = np.empty(self.size, dtype=np.int32)
        self.indices = np.empty(self.size, dtype=np.int32)
        self.algorithm = algorithm

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        if self.algorithm == ALLDIFFERENT_LOPEZ_ORTIZ:
            return alldifferent_lopez_ortiz_propagator.compute_domains(domains)
        if self.algorithm == SUM:
            return sum_propagator.compute_domains(domains)
        if self.algorithm == DUMMY:
            return dummy_propagator.compute_domains(domains)
        return None
