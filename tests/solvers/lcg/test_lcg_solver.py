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
import itertools
import random
from collections.abc import Callable
from functools import partial

import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.examples.golomb.golomb_problem import GolombProblem
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_DOM_WDEG,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_LEQ_C, ALG_LINEAR_LEQ_C, ALG_NEQ
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.choicepoints import (
    CHOICEPOINT_ITRAIL_MARK,
)
from nucs.solvers.lcg.implication_trail import REASON_DECISION, REASON_PROPAGATOR, REASON_ROOT
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.solvers.restart_policies import RESTART_LUBY, RestartPolicy
from nucs.solvers.searches import Search
from nucs.solvers.solver import OPTIM_PRUNE, OPTIM_RESET
from nucs.solvers.tree_search_solver import SOLVER_RUNNING, TreeSearchSolver
from nucs.statistics import (
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_ELAPSED_TIME,
    STATS_IDX_SOLVER_RESTART_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LCG_LBL_BACKJUMP_LEVEL_NB,
    STATS_LCG_LBL_CLAUSE_EXPLANATION_NB,
    STATS_LCG_LBL_CONFLICT_NB,
    STATS_LCG_LBL_LEARNED_LBD_SUM,
    STATS_LCG_LBL_LEARNED_LIT_NB,
    STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB,
)
from tests.propagators.propagator_test import random_bounds
from tests.solvers.backtrack.test_bc_algorithm import random_problem_with_copies


def run_queens_find_all(solver_class: Callable[..., TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    solver = solver_class(QueensProblem(7))
    return solver, [solution.tolist() for solution in solver.find_all()]


def run_queens_two_searches(solver_class: Callable[..., TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    # a sequential search: the first search branches on the first half of the queens, the second on the rest
    searches = [
        Search(decision_variables=range(4), var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN),
        Search(decision_variables=range(4, 8), dom_heuristic=DOM_HEURISTIC_SPLIT_LOW),
    ]
    solver = solver_class(QueensProblem(8), searches=searches)
    return solver, [solution.tolist() for solution in solver.find_all()]


def run_quasigroup_restarts(solver_class: Callable[..., TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    # dom/wdeg, last-conflict and restarts: the weights and the search control live across the descents
    solver = solver_class(
        QuasigroupProblem(5, 8, True),
        searches=[Search(var_heuristic=VAR_HEURISTIC_DOM_WDEG, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)],
        restart_policy=RestartPolicy(RESTART_LUBY, 2),  # 14 restarts in 101 choices
        last_conflict=True,
    )
    solution = next(solver.solve(), None)
    assert solver.statistics[STATS_IDX_SOLVER_RESTART_NB] > 0  # the case must exercise the restarts
    return solver, [None if solution is None else solution.tolist()]


class TestLcgSolver:
    @pytest.mark.parametrize(
        "run",
        [
            run_queens_find_all,
            run_queens_two_searches,
            lambda solver_class: run_golomb_find_best(solver_class, OPTIM_PRUNE),
            lambda solver_class: run_golomb_find_best(solver_class, OPTIM_RESET),
            run_quasigroup_restarts,
        ],
        ids=["find_all", "two_searches", "find_best_prune", "find_best_reset", "restarts_last_conflict"],
    )
    def test_same_answers_as_backtrack_solver(
        self, run: Callable[[Callable[..., TreeSearchSolver]], tuple[TreeSearchSolver, list]]
    ) -> None:
        # from step 10 of design/proposals/lcg.md 7.0, LcgSolver learns a clause at each conflict and jumps back: its
        # tree is not the tree of BacktrackSolver, but the solutions and the optima are the same (the order of the
        # solutions can change). The debug mode replays the implication trail and checks the explanations at each node:
        # restarts, the objective at the root and the objective after each backjump all write rows.
        _, backtrack_solutions = run(BacktrackSolver)
        _, lcg_solutions = run(partial(LcgSolver, debug=True))
        assert sorted(lcg_solutions) == sorted(backtrack_solutions)

    def test_run_search_loop_records_the_decisions_and_the_propagations(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(2))
        solver = LcgSolver(problem, debug=True)
        status, solution = solver._run_search_loop()
        assert status == SOLVER_RUNNING
        assert solution is not None
        assert solution.tolist() == [0, 1]
        # choice point 0 branches on the decision [x0 <= 0], at row 0; at choice point 1, alldifferent, the propagator
        # 0, then sets [x1 >= 1] in a call that starts at row 1
        assert solver.itrail_top.tolist() == [2]
        assert solver.itrail[:2].tolist() == [
            [0 << 1 | DOMAIN_MAX, 0, -1, 1, REASON_DECISION, -1, -1, -1],
            [1 << 1 | DOMAIN_MIN, 1, -1, 1, REASON_PROPAGATOR, 0, 1, 1],
        ]
        assert solver.choicepoints[0, CHOICEPOINT_ITRAIL_MARK] == 0
        assert solver._backtrack()
        # the nogood of the decisions, [x0 <= 0], gives the unit clause [x0 >= 1]: the search jumps back to level 0, and
        # the clause sets [x0 >= 1] there, for the rest of the search
        assert solver.choicepoint_top.tolist() == [0]
        assert solver.itrail_top.tolist() == [1]
        assert solver.itrail[0].tolist() == [0 << 1 | DOMAIN_MIN, 1, -1, 0, REASON_ROOT, -1, -1, -1]
        assert solver.itrail_last.tolist() == [0, -1, -1, -1]

    # below and at 14, the room of a call on 6 queens: 2 x 6, and the 2 rows of ITRAIL_STEP_ROW_NB
    @pytest.mark.parametrize("row_nb", [1, 13, 14])
    def test_find_all_the_implication_trail_grows_rather_than_overruns(self, row_nb: int) -> None:
        # a propagation that stops for room continues where it stopped: the counters are the same as without growth,
        # and the solutions are those of BacktrackSolver
        expected_solver = LcgSolver(QueensProblem(6), debug=True)
        expected_solutions = [solution.tolist() for solution in expected_solver.find_all()]
        assert expected_solutions == [solution.tolist() for solution in BacktrackSolver(QueensProblem(6)).find_all()]
        solver = LcgSolver(QueensProblem(6), debug=True)
        solver.itrail = solver.itrail[:row_nb].copy()
        assert [solution.tolist() for solution in solver.find_all()] == expected_solutions
        assert len(solver.itrail) > row_nb  # it did have to grow
        expected_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        assert solver.statistics.tolist() == expected_solver.statistics.tolist()

    def test_find_all_the_itrail_marks_grow_with_the_stack(self) -> None:
        # the marks of the implication trail are a column of the stack: a stack that grows keeps them; the stamps of
        # the levels, one for each choice point, grow with it
        backtrack_solutions = [solution.tolist() for solution in BacktrackSolver(QueensProblem(7)).find_all()]
        solver = LcgSolver(QueensProblem(7), debug=True)
        solver.choicepoints = solver.choicepoints[:4].copy()
        solver.level_stamps = solver.level_stamps[:4].copy()
        assert [solution.tolist() for solution in solver.find_all()] == backtrack_solutions
        assert len(solver.choicepoints) > 4  # it did have to grow
        assert len(solver.level_stamps) == len(solver.choicepoints)

    @pytest.mark.parametrize(
        "run",
        [
            lambda solver_class: run_queens_mid_value_find_all(solver_class),
            lambda solver_class: run_golomb_mid_value_find_best(solver_class),
        ],
        ids=["find_all", "find_best"],
    )
    def test_interior_eq_gives_the_same_answers_as_backtrack_solver(
        self, run: Callable[[Callable[..., TreeSearchSolver]], tuple[TreeSearchSolver, list]]
    ) -> None:
        # step 6 of design/proposals/lcg.md 7.0: an EQ inside the domain is two literals on two choice points, so the
        # tree changes, but not the solutions, nor the optimum
        backtrack_solver, backtrack_solutions = run(BacktrackSolver)
        lcg_solver, lcg_solutions = run(partial(LcgSolver, debug=True))
        assert sorted(lcg_solutions) == sorted(backtrack_solutions)
        assert lcg_solver.statistics[STATS_IDX_SOLUTION_NB] == backtrack_solver.statistics[STATS_IDX_SOLUTION_NB]
        # the case must exercise the second literals: a tree of binary choices has more choices than a ternary one
        assert (
            lcg_solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] > backtrack_solver.statistics[STATS_IDX_SOLVER_CHOICE_NB]
        )

    def test_run_search_loop_records_the_two_literals_of_an_interior_eq(self) -> None:
        problem = Problem([(0, 2)])
        searches = [Search(dom_heuristic=DOM_HEURISTIC_MID_VALUE)]
        solver = LcgSolver(problem, searches=searches, debug=True)
        status, solution = solver._run_search_loop()
        assert status == SOLVER_RUNNING
        assert solution is not None
        assert solution.tolist() == [1]
        # x0 = 1 is [x0 <= 1] at choice point 1, then [x0 >= 1] at choice point 2
        assert solver.itrail_top.tolist() == [2]
        assert solver.itrail[:2].tolist() == [
            [0 << 1 | DOMAIN_MAX, 1, -1, 1, REASON_DECISION, -1, -1, -1],
            [0 << 1 | DOMAIN_MIN, 1, -1, 2, REASON_DECISION, -1, -1, -1],
        ]
        # each choice point holds the row of the implication trail where its level ends
        assert solver.choicepoints[:2, CHOICEPOINT_ITRAIL_MARK].tolist() == [0, 1]
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == 2
        # the values come in the order of BacktrackSolver: the middle, then below it, then above it
        solutions = [solution.tolist() for solution in LcgSolver(problem, searches=searches, debug=True).solve()]
        assert solutions == [solution.tolist() for solution in BacktrackSolver(problem, searches=searches).solve()]
        assert solutions == [[1], [0], [2]]

    def test_run_search_loop_drops_a_second_literal_that_the_propagation_made_false(self) -> None:
        problem = Problem([(0, 4), (2, 2)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(2))
        searches = [Search(decision_variables=[0], dom_heuristic=DOM_HEURISTIC_MID_VALUE)]
        solver = LcgSolver(problem, searches=searches, debug=True)
        status, solution = solver._run_search_loop()
        assert status == SOLVER_RUNNING
        assert solution is not None
        assert solution.tolist() == [0, 2]
        # [x0 <= 2] at choice point 1; alldifferent then sets [x0 <= 1], so [x0 >= 2] is false and opens no choice
        # point; the heuristic then splits [0, 1] at its min: [x0 <= 0] at choice point 2
        assert solver.itrail_top.tolist() == [3]
        assert solver.itrail[:3].tolist() == [
            [0 << 1 | DOMAIN_MAX, 2, -1, 1, REASON_DECISION, -1, -1, -1],
            [0 << 1 | DOMAIN_MAX, 1, 0, 1, REASON_PROPAGATOR, 0, 1, 0],
            [0 << 1 | DOMAIN_MAX, 0, 1, 2, REASON_DECISION, -1, -1, -1],
        ]
        solutions = [solution.tolist() for solution in LcgSolver(problem, searches=searches, debug=True).solve()]
        assert solutions == [solution.tolist() for solution in BacktrackSolver(problem, searches=searches).solve()]

    def test_find_all_with_copies_against_brute_force(self) -> None:
        # a propagator with a variable twice gets the write-back of copies, which writes rows and can fail after some of
        # them: the debug mode checks these rows, and the explanation of the failure at the position of the call
        rng = random.Random(20261113)
        for _ in range(200):
            problem, bounds, is_solution = random_problem_with_copies(rng)
            expected = sorted(
                point for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]) if is_solution(point)
            )
            found = sorted(tuple(solution.tolist()) for solution in LcgSolver(problem, debug=True).find_all())
            assert found == expected, f"{bounds} {problem.propagators}"

    @pytest.mark.parametrize("mode", [None, OPTIM_PRUNE, OPTIM_RESET])
    def test_random_problems_against_brute_force(self, mode: str | None) -> None:
        # from step 10, the search depends on the clauses: a clause that removes a solution changes the answer. Small
        # random problems of several small constraints give short clauses, which apply again: enumerate their solutions,
        # or minimize x0 in both modes, and compare with brute force
        rng = random.Random(20261115)
        for _ in range(100):
            problem, bounds, is_solution = random_small_problem(rng)
            expected = sorted(
                point for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]) if is_solution(point)
            )
            solver = LcgSolver(problem, debug=True)
            if mode is None:
                found = sorted(tuple(solution.tolist()) for solution in solver.find_all())
                assert found == expected, f"{bounds} {problem.propagators}"
            else:
                solution = solver.find_best(0, DOMAIN_MIN, mode=mode)
                assert (solution is None) == (not expected), f"{bounds} {problem.propagators}"
                if solution is not None:
                    assert solution[0] == min(point[0] for point in expected), f"{bounds} {problem.propagators}"
                    assert tuple(solution.tolist()) in expected

    @pytest.mark.parametrize("mode", [OPTIM_PRUNE, OPTIM_RESET])
    def test_optimize_yields_only_improving_solutions(self, mode: str) -> None:
        # minimize y, with x1 <= y and x2 <= y, from the largest values of x1 and x2. The first solution has y = 3, and
        # its clause forbids only x1 = 3: the bound y <= 2, which the search applies again after each backjump, is what
        # forbids the solution x1 = 2, x2 = 3, y = 3 that follows
        problem = Problem([(0, 3), (0, 3), (0, 3)])
        x1, x2, y = range(3)
        problem.add_propagator(ALG_LEQ_C, [x1, y], [0])
        problem.add_propagator(ALG_LEQ_C, [x2, y], [0])
        searches = [
            Search(decision_variables=[x1, x2], dom_heuristic=DOM_HEURISTIC_MAX_VALUE),
            Search(decision_variables=[y]),
        ]
        solver = LcgSolver(problem, searches=searches, debug=True)
        objectives = [int(solution[y]) for solution in solver.optimize(y, DOMAIN_MIN, mode)]
        assert objectives == sorted(set(objectives), reverse=True)  # each one improves on the previous one
        assert objectives[-1] == 0

    def test_find_all_the_clause_store_grows(self) -> None:
        # a conflict whose clause has no room waits in pending_conflict while the store grows: the answers are the same
        expected = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(QueensProblem(7)).find_all())
        solver = LcgSolver(QueensProblem(7), debug=True)
        solver.clause_lits = solver.clause_lits[: len(solver.learned) + 1].copy()
        solver.clause_start = solver.clause_start[:3].copy()
        solver.watches = solver.watches[:4].copy()
        solver.unit_lits = solver.unit_lits[:1].copy()
        solver.clause_log_info = solver.clause_log_info[:2].copy()
        assert sorted(tuple(solution.tolist()) for solution in solver.find_all()) == expected
        assert len(solver.clause_start) > 3 and len(solver.clause_log_info) > 2  # they did have to grow

    def test_statistics_count_the_learned_clauses(self) -> None:
        # the counters of the learning agree with the clauses that the debug mode logs; queens 6 learns 36 clauses,
        # and its clauses have fewer levels than literals (an LBD sum of 70 for 162 literals)
        solver = LcgSolver(QueensProblem(6), debug=True)
        assert len(list(solver.find_all())) == 4
        statistics = solver.get_statistics_as_dictionary()
        clauses = solver.learned_clauses()
        literal_nb = sum(len(literals) for literals, _, _, _ in clauses)
        asserting_clause_nb = sum(1 for _, _, assertion_level, _ in clauses if assertion_level >= 0)
        assert statistics[STATS_LBL_SOLUTION_NB] == 4  # the statistics of TreeSearchSolver are still there
        assert statistics[STATS_LCG_LBL_CONFLICT_NB] == len(clauses) == 36
        assert statistics[STATS_LCG_LBL_LEARNED_LIT_NB] == literal_nb
        # an asserting clause has at least one level, and at most one for each literal
        assert asserting_clause_nb <= statistics[STATS_LCG_LBL_LEARNED_LBD_SUM] < literal_nb
        # a backjump leaves at least the levels from the conflict down to the assertion level
        assert statistics[STATS_LCG_LBL_BACKJUMP_LEVEL_NB] >= sum(
            level - assertion_level for _, level, assertion_level, _ in clauses if assertion_level >= 0
        )
        assert statistics[STATS_LCG_LBL_CLAUSE_EXPLANATION_NB] > 0
        # the problem has only alldifferent propagators
        explanation_nb = statistics[STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB]
        assert explanation_nb > 0
        assert statistics[f"{STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB}_ALLDIFFERENT"] == explanation_nb

    def test_debug_mode_finds_an_implication_trail_that_does_not_agree_with_the_state(self) -> None:
        solver = LcgSolver(QueensProblem(4), debug=True)
        solver.itrail_last[0] = 0  # the implication trail is empty, so no chain can end at a row
        with pytest.raises(AssertionError, match="at row 0 of 0"):
            solver._run_search_loop()


def run_golomb_find_best(solver_class: Callable[..., TreeSearchSolver], mode: str) -> tuple[TreeSearchSolver, list]:
    problem = GolombProblem(6)
    solver = solver_class(problem)
    solution = solver.find_best(problem.length_idx, DOMAIN_MIN, mode=mode)
    return solver, [None if solution is None else solution.tolist()]


def run_queens_mid_value_find_all(solver_class: Callable[..., TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    solver = solver_class(QueensProblem(7), searches=[Search(dom_heuristic=DOM_HEURISTIC_MID_VALUE)])
    return solver, [solution.tolist() for solution in solver.find_all()]


def run_golomb_mid_value_find_best(solver_class: Callable[..., TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    problem = GolombProblem(6)
    solver = solver_class(problem, searches=[Search(dom_heuristic=DOM_HEURISTIC_MID_VALUE)])
    solution = solver.find_best(problem.length_idx, DOMAIN_MIN)
    return solver, [None if solution is None else solution[problem.length_idx]]


def random_small_problem(
    rng: random.Random,
) -> tuple[Problem, list[tuple[int, int]], Callable[[tuple[int, ...]], bool]]:
    # five variables in small boxes, and four to seven constraints on two or three of them
    variable_nb = 5
    bounds = random_bounds(rng, variable_nb, 0, 3)
    problem = Problem(bounds)
    checks: list[Callable[[tuple[int, ...]], bool]] = []
    for _ in range(rng.randint(4, 7)):
        kind = rng.randrange(4)
        if kind == 0:
            x, y = rng.sample(range(variable_nb), 2)
            problem.add_propagator(ALG_NEQ, [x, y])
            checks.append(lambda point, x=x, y=y: point[x] != point[y])  # type: ignore[misc]
        elif kind == 1:
            x, y = rng.sample(range(variable_nb), 2)
            c = rng.randint(-2, 2)
            problem.add_propagator(ALG_LEQ_C, [x, y], [c])
            checks.append(lambda point, x=x, y=y, c=c: point[x] <= point[y] + c)  # type: ignore[misc]
        elif kind == 2:
            variables = rng.sample(range(variable_nb), 3)
            problem.add_propagator(ALG_ALLDIFFERENT, variables)
            checks.append(lambda point, vs=variables: len({point[v] for v in vs}) == 3)  # type: ignore[misc]
        else:
            variables = rng.sample(range(variable_nb), rng.randint(2, 3))
            coefficients = [rng.choice([-2, -1, 1, 2]) for _ in variables]
            c = rng.randint(-2, 5)
            problem.add_propagator(ALG_LINEAR_LEQ_C, variables, coefficients + [c])
            checks.append(
                lambda point, vs=variables, a=coefficients, c=c: sum(ai * point[v] for ai, v in zip(a, vs)) <= c  # type: ignore[misc]
            )
    return problem, bounds, lambda point: all(check(point) for check in checks)
