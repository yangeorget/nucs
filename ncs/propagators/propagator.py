import numpy as np
from numpy.typing import NDArray


class Propagator:
    def __init__(self, variables: NDArray, algorithm: int):
        self.size = len(variables)
        self.variables = variables
        self.triggers = np.ones((self.size, 2), dtype=np.bool)
        self.algorithm = algorithm
