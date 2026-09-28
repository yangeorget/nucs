###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_NEQ
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestProblem:
    def test_init_variable_propagators(self) -> None:
        problem = Problem([(0, 3)] * 4)
        problem.add_propagator(ALG_NEQ, [2, 0])
        problem.add_propagator(ALG_ALLDIFFERENT, [1, 2, 1])  # lists x1 twice
        problem.init()
        offsets = problem.variable_propagators_offsets.tolist()
        propagators = problem.variable_propagators.tolist()
        # x0 in p0, x1 in p1 once, x2 in both in increasing order, x3 in none
        assert [propagators[offsets[v] : offsets[v + 1]] for v in range(4)] == [[0], [1], [0, 1], []]

    def test_init_variable_propagators_without_propagators(self) -> None:
        problem = Problem([(0, 3)] * 2)
        problem.init()
        assert problem.variable_propagators_offsets.tolist() == [0, 0, 0]
        assert problem.variable_propagators.tolist() == []

    def test_init_twice_two_solvers_on_one_problem(self) -> None:
        # each solver initializes the problem again, which used to count the unbound variables twice: the second
        # solver never saw them all bound and found no solution
        problem = Problem([(0, 1), (0, 1)])
        problem.add_propagator(ALG_NEQ, [0, 1])
        assert len(BacktrackSolver(problem).find_all()) == 2
        assert problem.unbound_variable_nb == 2
        assert len(BacktrackSolver(problem).find_all()) == 2
        assert problem.unbound_variable_nb == 2
