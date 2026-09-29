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
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_LINEAR_LEQ_C, ALG_NEQ, watchers
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestProblem:
    def test_init_triggers_watchers(self) -> None:
        problem = Problem([(0, 3)] * 4)
        problem.add_propagator(ALG_NEQ, [2, 0])
        problem.add_propagator(ALG_ALLDIFFERENT, [1, 2, 1])  # lists x1 twice
        problem.add_propagator(ALG_LINEAR_LEQ_C, [3, 2], [0, 1, 3])  # 0 * x3 + x2 <= 3: never wakes on x3
        problem.init()
        rows = []
        for variable in range(4):
            start, end = watchers(problem.triggers_offsets, variable)
            rows.append(problem.triggers[start:end].tolist())
        # each watcher once, in increasing order; x3 is watched by nothing
        assert rows == [[0], [1], [0, 1, 2], []]

    def test_init_twice_two_solvers_on_one_problem(self) -> None:
        # each solver initializes the problem again, which used to count the unbound variables twice: the second
        # solver never saw them all bound and found no solution
        problem = Problem([(0, 1), (0, 1)])
        problem.add_propagator(ALG_NEQ, [0, 1])
        assert len(BacktrackSolver(problem).find_all()) == 2
        assert problem.unbound_variable_nb == 2
        assert len(BacktrackSolver(problem).find_all()) == 2
        assert problem.unbound_variable_nb == 2
