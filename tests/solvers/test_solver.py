from ncs.problems.problem import Problem
from ncs.propagators.sum import Sum
from ncs.solvers.solver import Solver


class TestSolver:
    def test_problem_filter(self) -> None:
        shr_domains = [(0, 2), (0, 2), (0, 6)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Sum([2, 0, 1]))
        solver = Solver(problem)
        problem.filter(None, solver.statistics)
        assert solver.statistics["problem.propagators.filters.nb"] == 1
