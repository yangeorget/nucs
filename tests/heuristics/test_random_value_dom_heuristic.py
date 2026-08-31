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
from nucs.heuristics.heuristics import DOM_HEURISTIC_RANDOM_VALUE
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestRandomValueDomHeuristic:
    def test_find_all(self) -> None:
        """Randomizing the order must not lose or repeat a solution.

        The order is what varies, so the order is the one thing not asserted: the enumeration is still a
        partition of the domain, because DECISION_EQ parks the values on either side of the one it draws.
        """
        problem = Problem([(1, 8)])
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_RANDOM_VALUE)
        solutions = solver.find_all()
        assert sorted(solution[0] for solution in solutions) == list(range(1, 9))

    def test_find_all_over_a_negative_domain(self) -> None:
        """A drawn value below zero has to survive the trip to the solver.

        random.randint hands back an int64 where the other domain heuristics return a domain bound that is
        already int32, so this is the one heuristic whose value is narrowed on the way out to
        SIGN_DOM_HEURISTIC. Without a negative domain the whole suite passes either way.
        """
        problem = Problem([(-5, -1)])
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_RANDOM_VALUE)
        solutions = solver.find_all()
        assert sorted(solution[0] for solution in solutions) == list(range(-5, 0))

    def test_find_all_over_several_variables(self) -> None:
        problem = Problem([(0, 2), (0, 2)])
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_RANDOM_VALUE)
        solutions = solver.find_all()
        assert sorted(tuple(solution) for solution in solutions) == [(x, y) for x in range(3) for y in range(3)]
