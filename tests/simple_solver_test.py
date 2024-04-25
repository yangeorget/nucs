import numpy as np

from ncs.constraints.sum import Sum
from ncs.problem import Problem
from ncs.solvers.simple_solver import SimpleSolver


class TestSimpleSolver:
    def test_solve_and_count(self) -> None:
        domains = np.array(
            [
                [0, 99],
                [0, 99]
            ]
        )
        problem = Problem(domains)
        solver = SimpleSolver(problem)
        counter = 0
        for _ in solver.solve():
            counter += 1
        assert counter == 10000

    def test_solve(self) -> None:
        domains = np.array(
            [
                [0, 1],
                [0, 1],
            ]
        )
        problem = Problem(domains)
        solver = SimpleSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 4
        assert np.all(solutions[0] == np.array([[0, 0], [0, 0]]))
        assert np.all(solutions[1] == np.array([[0, 0], [1, 1]]))
        assert np.all(solutions[2] == np.array([[1, 1], [0, 0]]))
        assert np.all(solutions[3] == np.array([[1, 1], [1, 1]]))

    def test_solve_sum_1(self) -> None:
        domains = np.array(
            [
                [0, 2],
                [0, 2],
                [4, 6],
            ]
        )
        problem = Problem(domains)
        problem.constraints.append(Sum(np.array([2, 0, 1])))
        solver = SimpleSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert np.all(solutions == np.array([[2, 2], [2, 2], [4, 4]]))

    def test_solve_sum_3(self) -> None:
        domains = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )
        problem = Problem(domains)
        problem.constraints.append(Sum(np.array([2, 0, 1])))
        solver = SimpleSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 3
        assert np.all(solutions[0] == np.array([[0, 0], [0, 0], [0, 0]]))
        assert np.all(solutions[1] == np.array([[0, 0], [1, 1], [1, 1]]))
        assert np.all(solutions[2] == np.array([[1, 1], [0, 0], [1, 1]]))

    def test_solve_sum_ko(self) -> None:
        domains = np.array(
            [
                [1, 2],
                [1, 2],
                [0, 1],
            ]
        )
        problem = Problem(domains)
        problem.constraints.append(Sum(np.array([2, 0, 1])))
        solver = SimpleSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 0
