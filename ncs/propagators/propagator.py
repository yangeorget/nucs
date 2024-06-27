from typing import List, Optional

import numba
import numpy as np
from numba.experimental import jitclass
from numpy.typing import NDArray

from ncs.propagators import alldifferent_lopez_ortiz, sum


@jitclass([
    ('size', numba.uint16),
    ('variables', numba.uint16[:]),
    ('triggers', numba.boolean[:,:]),
    ('offsets', numba.int32[:]),
    ('indices', numba.uint16[:]),
    ('name', numba.types.string)
])
class Propagator:
    def __init__(self, vars: List[int], name: str):
        self.size = len(vars)
        self.variables = np.array(vars, dtype=np.uint16)
        self.triggers = np.full((self.size, 2), True, dtype=bool)
        self.offsets = np.full((self.size, 1), 0, dtype=np.int32)
        self.indices = np.full((self.size, 1), 0, dtype=np.uint16)
        self.name = name

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        if self.name == "alldifferent_lopez_ortiz":
            return alldifferent_lopez_ortiz.compute_domains(domains)
        if self.name == "sum":
            return sum.compute_domains(domains)
        return None
