from ncs.problems.problem import ALG_SUM, Problem
from ncs.solvers.solver import Solver
from ncs.utils import STATS_PROBLEM_PROPAGATORS_FILTERS_NB


class TestSolver:
    def test_problem_filter(self) -> None:
        problem = Problem(shared_domains=[(0, 2), (0, 2), (0, 6)], domain_indices=[0, 1, 2], domain_offsets=[0, 0, 0])
        problem.set_propagators([([2, 0, 1], ALG_SUM, [])])
        solver = Solver(problem)
        problem.filter(solver.statistics)
        assert solver.statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] == 1
