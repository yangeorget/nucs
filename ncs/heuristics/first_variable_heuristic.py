from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problems.problem import Problem


class FirstVariableHeuristic(VariableHeuristic):
    """
    Chooses the first non instantiated variable.
    """

    def choose_variable(self, problem: Problem) -> int:
        for var_idx in range(problem.size):
            if problem.is_not_instantiated(var_idx):
                return var_idx
        return -1
