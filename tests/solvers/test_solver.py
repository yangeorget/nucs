import numpy as np

from ncs.problems.problem import ALGORITHM_SUM, Problem
from ncs.propagators.propagator import Propagator
from ncs.solvers.solver import Solver
from ncs.utils import STATS_PROBLEM_PROPAGATORS_FILTERS_NB


class TestSolver:
    def test_problem_filter(self) -> None:
        problem = Problem(shared_domains = [(0, 2), (0, 2), (0, 6)], domain_indices = [0, 1, 2], domain_offsets = [0, 0, 0])
        problem.set_propagators([([2, 0, 1], ALGORITHM_SUM)])
        solver = Solver(problem)
        problem.filter(solver.statistics)
        assert solver.statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] == 1
