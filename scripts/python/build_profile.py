import cProfile

import ncs.solvers.BacktrackSolver

cProfile.runctx("BacktrackSolver(QueensProblem(8)).solve_all()", globals(), locals(), "profile.pstats")
