from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problem import Problem


class FirstVariableHeuristic(VariableHeuristic):
    """
    Chooses the first non instantiated variable.
    """

    def make_variable_choice(self, problem: Problem) -> int:
        for idx in range(problem.domains.shape[0]):
            if not problem.is_instantiated(idx):
                return idx
        return -1
