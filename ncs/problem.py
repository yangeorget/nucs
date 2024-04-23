from typing import List

import numpy as np
from numpy.typing import NDArray

MIN = 0
MAX = 1


class Problem:
    def __init__(self, domains: NDArray):
        self.domains = domains
        self.constraints: List = []

    def filter(self) -> bool:
        print("filter()")
        changes = True
        while changes:
            changes = False
            for constraint in self.constraints:
                constraint_changes = constraint.filter(self)
                print(f"constraint_changes={constraint_changes}")
                if self.is_inconsistent():
                    return False
                changes |= np.any(constraint_changes)  # type: ignore
        return True

    def update_domains(self, variables: NDArray, new_domains: NDArray) -> NDArray:
        print(f"update_domains({variables}, {new_domains})")
        changes = np.full((len(new_domains), 2), False)
        new_minimums = np.maximum(new_domains[:, MIN], self.domains[variables, MIN])
        np.greater(new_minimums, self.domains[variables, MIN], out=changes[:, MIN])
        self.domains[variables, MIN] = new_minimums
        new_maximums = np.minimum(new_domains[:, MAX], self.domains[variables, MAX])
        np.less(new_maximums, self.domains[variables, MAX], out=changes[:, MAX])
        self.domains[variables, MAX] = new_maximums
        return changes

    def is_instantiated(self, idx: int) -> bool:
        print("is_instantiated()")
        return self.domains[idx, MIN] == self.domains[idx, MAX]

    def is_inconsistent(self) -> bool:
        print("is_inconsistent()")
        return np.any(np.greater(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def is_solved(self) -> bool:
        print("is_solved()")
        return np.all(np.equal(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def __str__(self) -> str:
        return f"domains={self.domains}, constraints={self.constraints}"
