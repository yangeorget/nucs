from cProfile import Profile
from pstats import SortKey, Stats

from ncs.problems.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver


def queens8() -> int:
    for _ in BacktrackSolver(QueensProblem(8)).solve():
        pass
    return 0


with Profile() as profile:
    print(f"{queens8()}")(Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats())
