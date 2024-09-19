from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_CHOICE_DEPTH, STATS_SOLVER_SOLUTION_NB


class TestBacktrackSolver:
    def test_solve_and_count(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 10000
        assert solver.statistics[STATS_SOLVER_CHOICE_DEPTH] == 2

    def test_solve(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 4
        assert solutions[0] == [0, 0]
        assert solutions[1] == [0, 1]
        assert solutions[2] == [1, 0]
        assert solutions[3] == [1, 1]
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 4
        assert solver.statistics[STATS_SOLVER_CHOICE_DEPTH] == 2

    def test_solve_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(([0, 1, 2], ALG_ALLDIFFERENT, []))
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 6
        assert solutions[0] == [0, 1, 2]
        assert solutions[1] == [0, 2, 1]
        assert solutions[2] == [1, 0, 2]
        assert solutions[3] == [1, 2, 0]
        assert solutions[4] == [2, 0, 1]
        assert solutions[5] == [2, 1, 0]
