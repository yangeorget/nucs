from numpy.typing import NDArray

from ncs.problem import Problem


class Constraint:
    def __init__(self, variables: NDArray):
        self.variables = variables

    def filter(self, problem: Problem) -> NDArray:
        print(f"filter({problem})")
        new_domains = self.compute_domains(problem)
        print(f"new_domains={new_domains}")
        return problem.update_domains(self.variables, new_domains)

    def compute_domains(self, problem: Problem) -> NDArray:  # type: ignore
        pass
