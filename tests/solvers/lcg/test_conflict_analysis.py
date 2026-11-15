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

import numpy as np
import pytest
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, LOG_LEVEL_ERROR, PROP_CONSISTENCY
from nucs.examples.golomb.golomb_problem import GolombProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_DOM_WDEG,
)
from nucs.problems.problem import Problem
from nucs.propagators.explanations import EXPLAIN_SIDE, EXPLAIN_VALUE, EXPLAIN_VAR_IDX
from nucs.propagators.propagators import ALG_LEQ_C, ALG_LEQ_C_IMP, ALG_LINEAR_LEQ_C, register_propagator
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.conflict_analysis import MARK_NONE, MARK_ROW, analyze_conflict
from nucs.solvers.lcg.implication_trail import (
    REASON_DECISION,
    REASON_PROPAGATOR,
    REASON_ROOT,
    allocate_itrail,
    itrail_push,
)
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.solvers.restart_policies import RESTART_LUBY, RestartPolicy
from nucs.solvers.searches import Search
from nucs.solvers.solver import OPTIM_PRUNE, OPTIM_RESET
from tests.marks import SLOW_WITHOUT_JIT
from tests.solvers.backtrack.test_bc_algorithm import random_problem_with_copies

# A row of a hand-built implication trail: (var_side, value, choicepoint, reason_kind, reason_data, reason_pos,
# reason_var_idx).
Row = tuple[int, int, int, int, int, int, int]
# A literal of an explanation, as the parameters of ALG_GIVEN hold it: (var_idx, side, value).
Given = tuple[int, int, int]


@njit(cache=True)
def compute_domains_given(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    return PROP_CONSISTENCY


@njit(cache=True)
def explain_given(domains: NDArray, parameters: NDArray, var_idx: int, side: int, value: int, out: NDArray) -> int:
    # the explanation of each row and of the failure is the list of literals in the parameters, three cells each
    lit_nb = len(parameters) // 3
    for lit in range(lit_nb):
        out[lit, EXPLAIN_VAR_IDX] = parameters[3 * lit]
        out[lit, EXPLAIN_SIDE] = parameters[3 * lit + 1]
        out[lit, EXPLAIN_VALUE] = parameters[3 * lit + 2]
    return lit_nb


@njit(cache=True)
def get_triggers_given(n: int, variable: int, parameters: NDArray) -> int:
    return EVENT_MASK_MIN_MAX


def get_complexity_given(n: int, parameters: NDArray) -> int:
    return n


# A propagator for the hand-built implication trails: it changes nothing, and its explanation is in its parameters, so
# that a test sets the explanation of each row and of the failure.
ALG_GIVEN = register_propagator(
    get_triggers_given, get_complexity_given, compute_domains_given, explain_fct=explain_given
)


def ge(variable: int, value: int) -> Given:
    return variable, DOMAIN_MIN, value


def decision(variable: int, side: int, value: int, choicepoint: int) -> Row:
    return (variable << 1) | side, value, choicepoint, REASON_DECISION, -1, -1, -1


def root(variable: int, side: int, value: int, choicepoint: int) -> Row:
    return (variable << 1) | side, value, choicepoint, REASON_ROOT, -1, -1, -1


def propagation(variable: int, side: int, value: int, choicepoint: int, prop_idx: int, pos: int) -> Row:
    # the propagators of these trails have the variables 0 to 3, in this order: the index of x_i is i
    return (variable << 1) | side, value, choicepoint, REASON_PROPAGATOR, prop_idx, pos, variable


def le_clause(variable: int, value: int) -> tuple[int, int]:
    return (variable << 1) | DOMAIN_MAX, value


class TestConflictAnalysis:
    def test_analyze_conflict_worked_example(self) -> None:
        # the example of design/proposals/lcg.md 1.11, with the naive explanations: C8 fails with [x >= 2] and [u >= 3],
        # whose reasons both reduce to [v >= 1], the first UIP; the learned clause [v <= 0] asserts at level 0, past the
        # decision on y at level 1
        problem, (w, v, x, u, y) = worked_example_problem()
        rows = [
            decision(y, DOMAIN_MAX, 4, 1),
            decision(w, DOMAIN_MIN, 1, 2),
            ((v << 1) | DOMAIN_MIN, 1, 2, REASON_PROPAGATOR, 0, 2, 1),  # C5 on [w, v]
            ((x << 1) | DOMAIN_MIN, 2, 2, REASON_PROPAGATOR, 1, 3, 2),  # C6 on [v, k, x]
            ((u << 1) | DOMAIN_MIN, 3, 2, REASON_PROPAGATOR, 2, 4, 2),  # C7 on [v, k, u]
        ]
        assert analyze(problem, rows, 3) == ([le_clause(v, 0)], 2, 0)

    @pytest.mark.parametrize(
        "rows, explanations, nogood, expected",
        [
            # the row of [x0 >= 2] is the first row that makes it true, at level 1, not the last change of x0 at level 2:
            # the clause keeps [x0 <= 1], and asserts at level 1
            (
                [decision(0, DOMAIN_MIN, 2, 1), decision(1, DOMAIN_MIN, 1, 2), propagation(0, DOMAIN_MIN, 4, 2, 0, 2)],
                [[ge(1, 1)]],
                [ge(0, 2), ge(1, 1)],
                ([le_clause(1, 0), le_clause(0, 1)], 2, 1),
            ),
            # [x0 >= 2] and [x0 >= 3] merge into [x0 >= 3], in both orders: one literal of level 2, the UIP
            (
                [decision(0, DOMAIN_MIN, 2, 1), decision(1, DOMAIN_MIN, 1, 2), propagation(0, DOMAIN_MIN, 3, 2, 0, 2)],
                [[ge(1, 1)]],
                [ge(0, 2), ge(0, 3)],
                ([le_clause(0, 2)], 2, 0),
            ),
            (
                [decision(0, DOMAIN_MIN, 2, 1), decision(1, DOMAIN_MIN, 1, 2), propagation(0, DOMAIN_MIN, 3, 2, 0, 2)],
                [[ge(1, 1)]],
                [ge(0, 3), ge(0, 2)],
                ([le_clause(0, 2)], 2, 0),
            ),
            # the explanations of two rows of level 2 meet at the decision of level 2, and bring a literal of level 1
            (
                [
                    decision(0, DOMAIN_MIN, 1, 1),
                    decision(1, DOMAIN_MIN, 1, 2),
                    propagation(2, DOMAIN_MIN, 1, 2, 0, 2),
                    propagation(3, DOMAIN_MIN, 1, 2, 1, 3),
                ],
                [[ge(0, 1), ge(1, 1)], [ge(1, 1)]],
                [ge(2, 1), ge(3, 1)],
                ([le_clause(1, 0), le_clause(0, 0)], 2, 1),
            ),
            # no literal of the nogood is at the current level 2: the conflict is at level 1
            (
                [decision(0, DOMAIN_MIN, 1, 1), decision(1, DOMAIN_MIN, 1, 2)],
                [],
                [ge(0, 1)],
                ([le_clause(0, 0)], 1, 0),
            ),
            # a level with two decisions, as after a backtrack to it: they have no explanation and both stay, so the
            # clause does not assert a literal
            (
                [decision(0, DOMAIN_MIN, 1, 1), decision(1, DOMAIN_MIN, 1, 1)],
                [],
                [ge(0, 1), ge(1, 1)],
                ([le_clause(0, 0), le_clause(1, 0)], 1, -1),
            ),
            # a literal of level 0 and a literal with REASON_ROOT hold for the rest of the search: they are dropped
            (
                [decision(0, DOMAIN_MIN, 1, 0), root(2, DOMAIN_MIN, 5, 1), decision(1, DOMAIN_MIN, 1, 1)],
                [],
                [ge(0, 1), ge(2, 5), ge(1, 1)],
                ([le_clause(1, 0)], 1, 0),
            ),
            # all the literals hold at level 0: the clause is empty, and no solution is left
            (
                [decision(0, DOMAIN_MIN, 1, 0)],
                [],
                [ge(0, 1)],
                ([], 0, -1),
            ),
        ],
        ids=[
            "first_row",
            "merge_stronger_last",
            "merge_stronger_first",
            "lower_level_literal",
            "no_literal_of_the_current_level",
            "two_decisions_on_a_level",
            "level_0_and_root_dropped",
            "empty_clause",
        ],
    )
    def test_analyze_conflict(
        self,
        rows: list[Row],
        explanations: list[list[Given]],
        nogood: list[Given],
        expected: tuple[list[tuple[int, int]], int, int],
    ) -> None:
        problem = Problem([(0, 9)] * 4)
        for explanation in [*explanations, nogood]:
            problem.add_propagator(ALG_GIVEN, range(4), [cell for literal in explanation for cell in literal])
        assert analyze(problem, rows, len(explanations)) == expected

    def test_run_search_loop_learns_the_clause_of_the_worked_example(self) -> None:
        # the search of design/proposals/lcg.md 1.11, except that the search splits y until it is fixed: [y <= 4],
        # [y <= 2], [y <= 1] and [y <= 0] on levels 1 to 4, then [w >= 1] on level 5. The first failure gives [v <= 0],
        # which asserts at level 0, past the four decisions on y. The third search branches on the other variables, so
        # that the search can end on a solution.
        problem, (w, v, x, u, y) = worked_example_problem()
        searches = [
            Search(decision_variables=[y], dom_heuristic=DOM_HEURISTIC_SPLIT_LOW),
            Search(decision_variables=[w], dom_heuristic=DOM_HEURISTIC_MAX_VALUE),
            Search(decision_variables=[v, x, u]),
        ]
        solver = LcgSolver(problem, searches=searches, debug=True, log_level=LOG_LEVEL_ERROR)
        assert next(solver.solve(), None) is not None
        assert solver.learned_clauses()[0] == ([le_clause(v, 0)], 5, 0, 0)

    @pytest.mark.parametrize(
        "run",
        [
            lambda: run_find_all(QueensProblem(6)),
            # the cases below run the same analysis; without the JIT, the brute force of their solutions takes 5 s each
            pytest.param(lambda: run_first_solution_with_restarts(QueensProblem(10)), marks=SLOW_WITHOUT_JIT),
            # from the largest values: 28 improving solutions, and more failures than the default search
            pytest.param(
                lambda: run_optimize(OPTIM_PRUNE, [Search(dom_heuristic=DOM_HEURISTIC_MAX_VALUE)]),
                marks=SLOW_WITHOUT_JIT,
            ),
            pytest.param(
                lambda: run_optimize(OPTIM_RESET, [Search(dom_heuristic=DOM_HEURISTIC_MAX_VALUE)]),
                marks=SLOW_WITHOUT_JIT,
            ),
            pytest.param(
                lambda: run_optimize(OPTIM_PRUNE, [Search(dom_heuristic=DOM_HEURISTIC_MID_VALUE)]),
                marks=SLOW_WITHOUT_JIT,
            ),
        ],
        ids=["find_all", "first_solution_restarts", "optimize_prune", "optimize_reset", "optimize_interior_eq"],
    )
    def test_learned_clauses_are_valid(
        self, run: Callable[[], tuple[LcgSolver, list[list[int]], list[list[int]], Callable]]
    ) -> None:
        # step 8 of design/proposals/lcg.md 7.0 and the clause test of 7.2: each clause keeps each solution that the
        # search has not found yet (an enumeration), or that is better than the best one (an optimization)
        solver, found, solutions, is_kept = run()
        clauses = solver.learned_clauses()
        assert any(assertion_level >= 0 for _, _, assertion_level, _ in clauses)  # the case must learn clauses
        assert_clauses_keep_the_solutions(clauses, found, np.array(solutions), is_kept)

    def test_learned_clauses_with_copies_are_valid(self) -> None:
        # a propagator with a variable twice writes the copies in one variable, and can fail after it wrote rows
        rng = random.Random(20261115)
        clause_nb = 0
        for _ in range(100):
            problem, bounds, is_solution = random_problem_with_copies(rng)
            solutions = [
                list(point)
                for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds])
                if is_solution(point)
            ]
            solver = LcgSolver(problem, debug=True, log_level=LOG_LEVEL_ERROR)
            found = [solution.tolist() for solution in solver.find_all()]
            clauses = solver.learned_clauses()
            clause_nb += len(clauses)
            assert_clauses_keep_the_solutions(
                clauses, found, np.array(solutions, dtype=np.int64).reshape(-1, len(bounds)), not_found_yet
            )
        assert clause_nb > 0


def analyze(problem: Problem, rows: list[Row], failed_prop_idx: int) -> tuple[list[tuple[int, int]], int, int]:
    solver = LcgSolver(problem, log_level=LOG_LEVEL_ERROR)
    itrail, itrail_top, itrail_last = allocate_itrail(problem.domain_nb)
    itrail_sz = 0
    for var_side, value, choicepoint, reason_kind, reason_data, reason_pos, reason_var_idx in rows:
        itrail_sz = itrail_push(
            itrail,
            itrail_last,
            itrail_sz,
            var_side,
            value,
            choicepoint,
            reason_kind,
            reason_data,
            reason_pos,
            reason_var_idx,
        )
    itrail_top[0] = itrail_sz
    lit_nb, level, assertion_level = analyze_conflict(
        solver.explain_addrs,
        problem.propagator_infos,
        problem.offsets,
        problem.propagator_variables,
        problem.propagator_parameters,
        problem.initial_domains,
        itrail,
        itrail_top,
        itrail_last,
        failed_prop_idx,
        itrail_sz,
        solver.domain_buffer,
        solver.explanation,
        solver.explanation_lits,
        solver.marks,
        solver.marked,
        solver.learned,
    )
    assert (solver.marks[:, MARK_ROW] == MARK_NONE).all()  # the analysis clears its marks
    return [(var_side, value) for var_side, value in solver.learned[:lit_nb].tolist()], level, assertion_level


def worked_example_problem() -> tuple[Problem, tuple[int, int, int, int, int]]:
    # w, v in {0, 1}; x, u in [0, 5]; y in [0, 9]; k = 2 holds the constant of the half-reified comparisons
    w, v, x, u, y, k = range(6)
    problem = Problem([(0, 1), (0, 1), (0, 5), (0, 5), (0, 9), (2, 2)])
    problem.add_propagator(ALG_LEQ_C, [w, v], [0])  # C5: w <= v
    problem.add_propagator(ALG_LEQ_C_IMP, [v, k, x], [0])  # C6: v = 1 -> 2 <= x
    problem.add_propagator(ALG_LEQ_C_IMP, [v, k, u], [-1])  # C7: v = 1 -> 2 <= u - 1
    problem.add_propagator(ALG_LINEAR_LEQ_C, [x, u], [1, 1, 4])  # C8: x + u <= 4
    return problem, (w, v, x, u, y)


def all_solutions(problem: Problem) -> list[list[int]]:
    return [solution.tolist() for solution in BacktrackSolver(problem, log_level=LOG_LEVEL_ERROR).find_all()]


def not_found_yet(solutions: NDArray, found: list[list[int]], solution_nb: int) -> NDArray:
    # the solutions that the search had not found when it learned the clause
    kept = np.ones(len(solutions), dtype=bool)
    for solution in found[:solution_nb]:
        kept &= (solutions != np.array(solution)).any(axis=1)
    return kept


def run_find_all(problem: Problem) -> tuple[LcgSolver, list[list[int]], list[list[int]], Callable]:
    solutions = all_solutions(problem)
    solver = LcgSolver(problem, debug=True, log_level=LOG_LEVEL_ERROR)
    found = [solution.tolist() for solution in solver.find_all()]
    assert sorted(found) == sorted(solutions)
    return solver, found, solutions, not_found_yet


def run_first_solution_with_restarts(problem: Problem) -> tuple[LcgSolver, list[list[int]], list[list[int]], Callable]:
    # dom/wdeg, last-conflict and restarts: the restarts build the root of the implication trail again
    solutions = all_solutions(problem)
    solver = LcgSolver(
        problem,
        searches=[Search(var_heuristic=VAR_HEURISTIC_DOM_WDEG, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)],
        restart_policy=RestartPolicy(RESTART_LUBY, 2),
        last_conflict=True,
        debug=True,
        log_level=LOG_LEVEL_ERROR,
    )
    solution = next(solver.solve())
    return solver, [solution.tolist()], solutions, not_found_yet


def run_optimize(mode: str, searches: list[Search]) -> tuple[LcgSolver, list[list[int]], list[list[int]], Callable]:
    problem = GolombProblem(5)
    length_idx = problem.length_idx
    solutions = all_solutions(GolombProblem(5))
    solver = LcgSolver(problem, searches=searches, debug=True, log_level=LOG_LEVEL_ERROR)
    found = [solution.tolist() for solution in solver.optimize(length_idx, DOMAIN_MIN, mode)]
    assert found[-1][length_idx] == min(solution[length_idx] for solution in solutions)

    def better_than_best(solutions: NDArray, found: list[list[int]], solution_nb: int) -> NDArray:
        # the solutions that are better than the best one when the search learned the clause
        if solution_nb == 0:
            return np.ones(len(solutions), dtype=bool)
        return solutions[:, length_idx] < found[solution_nb - 1][length_idx]

    return solver, found, solutions, better_than_best


def assert_clauses_keep_the_solutions(
    clauses: list[tuple[list[tuple[int, int]], int, int, int]],
    found: list[list[int]],
    solutions: NDArray,
    is_kept: Callable[[NDArray, list[list[int]], int], NDArray],
) -> None:
    for literals, level, assertion_level, solution_nb in clauses:
        assert assertion_level < level or assertion_level == -1 or not literals
        satisfied = np.zeros(len(solutions), dtype=bool)
        for var_side, value in literals:
            column = solutions[:, var_side >> 1]
            satisfied |= column >= value if (var_side & 1) == DOMAIN_MIN else column <= value
        violated = is_kept(solutions, found, solution_nb) & ~satisfied
        assert not violated.any(), f"{literals} removes {solutions[violated][0].tolist()}"
