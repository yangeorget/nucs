from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problem import Problem


class FirstVariableHeuristic(VariableHeuristic):
    def make_variable_choice(self, problem: Problem) -> int:
        # print("makeChoice()")
        for idx in range(problem.domains.shape[0]):
            if not problem.is_instantiated(idx):
                return idx
        return -1
