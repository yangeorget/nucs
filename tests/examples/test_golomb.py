from ncs.problems.golomb_problem import GolombProblem, index, init_domains
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import MIN


class TestGolomb:

    def test_index(self) -> None:
        assert index(4, 0, 1) == 0
        assert index(4, 0, 2) == 1
        assert index(4, 0, 3) == 2
        assert index(4, 1, 2) == 3
        assert index(4, 1, 3) == 4
        assert index(4, 2, 3) == 5

    def test_init_domains(self) -> None:
        domains = init_domains(6, 4)
        assert domains[:, MIN].tolist() == [1, 3, 6, 1, 3, 1]

    def test_golomb_4(self) -> None:
        problem = GolombProblem(4)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.length)
        assert solution
        assert solution[problem.length] == 6

    def test_golomb_5(self) -> None:
        problem = GolombProblem(5)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.length)
        assert solution
        assert solution[problem.length] == 11

    def test_golomb_6(self) -> None:
        problem = GolombProblem(6)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.length)
        assert solution
        assert solution[problem.length] == 17

    def test_golomb_7(self) -> None:
        problem = GolombProblem(7)
        solver = BacktrackSolver(problem)
        solution = solver.minimize(problem.length)
        assert solution
        assert solution[problem.length] == 25


if __name__ == "__main__":
    problem = GolombProblem(10)
    solver = BacktrackSolver(problem)
    solution = solver.minimize(problem.length)
    print(solution[problem.length])  # type: ignore
