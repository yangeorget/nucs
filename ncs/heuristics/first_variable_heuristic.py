from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problems.problem import Problem, not_instantiated_index


class FirstVariableHeuristic(VariableHeuristic):
    """
    Chooses the first non instantiated variable.
    """

    def choose_variable(self, problem: Problem) -> int:
        return not_instantiated_index(problem.variable_nb, problem.shared_domains, problem.domain_indices)
