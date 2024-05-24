from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problem import MAX, MIN, Problem


class FirstVariableHeuristic(VariableHeuristic):
    """
    Chooses the first non instantiated variable.
    """

    def choose_variable(self, problem: Problem) -> int:
        for idx in range(problem.domains.shape[0]):
            if problem.domains[idx, MIN] < problem.domains[idx, MAX]:
                return idx
        return -1
