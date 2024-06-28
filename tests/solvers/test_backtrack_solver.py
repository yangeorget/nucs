import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.propagator import Propagator
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestBacktrackSolver:
    def test_solve_and_count(self) -> None:
        shr_domains = [(0, 99), (0, 99)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([], dtype=np.int32), "dummy"))
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 10000
        assert solver.statistics["solver.cp.max"] == 2

    def test_solve(self) -> None:
        shr_domains = [(0, 1), (0, 1)]
        dom_indices = [0, 1]
        dom_offsets = [0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([], dtype=np.int32), "dummy"))
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 4
        assert solutions[0] == [0, 0]
        assert solutions[1] == [0, 1]
        assert solutions[2] == [1, 0]
        assert solutions[3] == [1, 1]
        assert solver.statistics["solver.solutions.nb"] == 4
        assert solver.statistics["solver.cp.max"] == 2

    def test_solve_sum_1(self) -> None:
        shr_domains = [(0, 2), (0, 2), (4, 6)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([2, 0, 1], dtype=np.int32), "sum"))
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert solutions == [[2, 2, 4]]
        assert solver.statistics["solver.solutions.nb"] == 1
        assert solver.statistics["problem.filters.nb"] == 1
        assert solver.statistics["solver.cp.max"] == 0
        assert solver.statistics["solver.backtracks.nb"] == 0

    def test_solve_sum_3(self) -> None:
        shr_domains = [(0, 1), (0, 1), (0, 1)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([2, 0, 1], dtype=np.int32), "sum"))
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 3
        assert solutions[0] == [0, 0, 0]
        assert solutions[1] == [0, 1, 1]
        assert solutions[2] == [1, 0, 1]
        assert solver.statistics["solver.solutions.nb"] == 3
        assert solver.statistics["solver.cp.max"] == 2

    def test_solve_sum_ko(self) -> None:
        shr_domains = [(1, 2), (1, 2), (0, 1)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([2, 0, 1], dtype=np.int32), "sum"))
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 0
        assert solver.statistics["problem.filters.nb"] == 1
        assert solver.statistics["solver.cp.max"] == 0

    def test_solve_alldifferent(self) -> None:
        shr_domains = [(0, 2), (0, 2), (0, 2)]
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        problem.add_propagator(Propagator(np.array([0, 1, 2], dtype=np.int32), "alldifferent_lopez_ortiz"))
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 6
        assert solutions[0] == [0, 1, 2]
        assert solutions[1] == [0, 2, 1]
        assert solutions[2] == [1, 0, 2]
        assert solutions[3] == [1, 2, 0]
        assert solutions[4] == [2, 0, 1]
        assert solutions[5] == [2, 1, 0]
