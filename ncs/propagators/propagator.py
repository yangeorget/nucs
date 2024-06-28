from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.propagators import alldifferent_lopez_ortiz_propagator, sum_propagator, dummy_propagator


# @jitclass(
#     [
#         ("size", numba.int32),
#         ("variables", numba.int32[:]),
#         ("triggers", numba.boolean[:, :]),
#         ("offsets", numba.int32[:]),
#         ("indices", numba.int32[:]),
#         ("name", numba.types.string),
#     ]
# )
class Propagator:
    def __init__(self, variables: NDArray, name: str):
        self.size = len(variables)
        self.variables = variables
        self.triggers = np.ones((self.size, 2), dtype=np.bool)
        self.offsets = np.empty(self.size, dtype=np.int32)
        self.indices = np.empty(self.size, dtype=np.int32)
        self.name = name

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        if self.name == "alldifferent_lopez_ortiz":
            return alldifferent_lopez_ortiz_propagator.compute_domains(domains)
        if self.name == "sum":
            return sum_propagator.compute_domains(domains)
        if self.name == "dummy":
            return dummy_propagator.compute_domains(domains)
        return None
