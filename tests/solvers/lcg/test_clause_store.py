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
import numpy as np

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, LOG_LEVEL_ERROR
from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.problems.problem import Problem
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.clause_store import (
    CLAUSES_ITRAIL_FULL,
    CLAUSES_PROPAGATED,
    STORE_CLAUSE_NB,
    STORE_UNIT_NB,
    WATCH_NEXT,
    attach_clause,
    propagate_clauses,
    store_clause,
    store_unit,
)
from nucs.solvers.lcg.implication_trail import (
    ITRAIL_REASON_DATA,
    ITRAIL_REASON_KIND,
    ITRAIL_VALUE,
    ITRAIL_VAR_SIDE,
    REASON_CLAUSE,
    REASON_DECISION,
    REASON_ROOT,
    itrail_push,
)
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.statistics import STATS_IDX_SOLUTION_NB, STATS_IDX_SOLVER_CHOICE_NB


def ge(variable: int, value: int) -> tuple[int, int]:
    return (variable << 1) | DOMAIN_MIN, value


def le(variable: int, value: int) -> tuple[int, int]:
    return (variable << 1) | DOMAIN_MAX, value


class TestClauseStore:
    def test_attach_clause_watches_two_literals_that_are_not_false(self) -> None:
        solver = new_solver(3)
        clause, attached = add_clause(solver, [ge(0, 5), le(1, 3), ge(2, 1)])
        assert attached
        assert watched(solver, clause) == [ge(0, 5), le(1, 3)]
        # [x >= v] is watched on the upper bound of x, [x <= v] on its lower bound
        assert watches_of(solver, le(0, 0)[0]) == [clause << 1]
        assert watches_of(solver, ge(1, 0)[0]) == [(clause << 1) | 1]

    def test_attach_clause_sets_the_last_literal_that_is_not_false(self) -> None:
        # x0 <= 4 makes [x0 >= 5] false, so the clause sets [x1 <= 3]
        solver = new_solver(2)
        set_bound(solver, le(0, 4), 1)
        clause, attached = add_clause(solver, [ge(0, 5), le(1, 3)])
        assert attached
        assert solver.domains[1].tolist() == [0, 3]
        assert last_row(solver) == (le(1, 3), REASON_CLAUSE, clause)

    def test_attach_clause_watches_the_literals_that_became_false_last(self) -> None:
        # one literal is free: the second watch is the false literal whose row is the highest, [x2 >= 5] here
        solver = new_solver(3)
        set_bound(solver, le(1, 4), 1)
        set_bound(solver, le(2, 4), 2)
        clause, attached = add_clause(solver, [ge(1, 5), ge(2, 5), le(0, 3)])
        assert attached
        assert watched(solver, clause) == [le(0, 3), ge(2, 5)]

    def test_attach_clause_reports_a_clause_whose_literals_are_all_false(self) -> None:
        # the level violates the clause: the call returns false, and the clause still gets its watches, on the two
        # literals that became false last ([x1 <= 3] at row 1, then [x0 >= 5] at row 0), so that it never stays unwatched
        solver = new_solver(2)
        set_bound(solver, le(0, 4), 1)
        set_bound(solver, ge(1, 4), 1)
        clause, attached = add_clause(solver, [ge(0, 5), le(1, 3)])
        assert not attached
        assert watched(solver, clause) == [le(1, 3), ge(0, 5)]
        assert watches_of(solver, ge(1, 0)[0]) == [clause << 1]
        assert watches_of(solver, le(0, 0)[0]) == [(clause << 1) | 1]

    def test_propagate_clauses_sets_the_last_literal(self) -> None:
        solver = new_solver(2)
        clause, _ = add_clause(solver, [ge(0, 5), le(1, 3)])
        set_bound(solver, le(0, 4), 1)
        assert propagate(solver) == CLAUSES_PROPAGATED
        assert solver.domains[1].tolist() == [0, 3]
        assert last_row(solver) == (le(1, 3), REASON_CLAUSE, clause)
        assert solver.clause_head[0] == solver.itrail_top[0]  # all the rows are visited

    def test_propagate_clauses_moves_a_watch_to_a_literal_that_is_not_false(self) -> None:
        solver = new_solver(3)
        clause, _ = add_clause(solver, [ge(0, 5), le(1, 3), ge(2, 1)])
        set_bound(solver, le(0, 4), 1)
        assert propagate(solver) == CLAUSES_PROPAGATED
        assert solver.domains.tolist() == [[0, 4], [0, 9], [0, 9]]  # nothing to set: two literals are free
        assert watched(solver, clause) == [ge(2, 1), le(1, 3)]
        assert watches_of(solver, le(0, 0)[0]) == []
        assert watches_of(solver, le(2, 0)[0]) == [clause << 1]

    def test_propagate_clauses_keeps_the_watch_of_a_clause_that_holds(self) -> None:
        # [x1 <= 3] is true: the clause holds, and the watch of the false [x0 >= 5] stays
        solver = new_solver(2)
        clause, _ = add_clause(solver, [ge(0, 5), le(1, 3)])
        set_bound(solver, le(1, 2), 1)
        set_bound(solver, le(0, 4), 1)
        assert propagate(solver) == CLAUSES_PROPAGATED
        assert solver.domains.tolist() == [[0, 4], [0, 2]]
        assert watches_of(solver, le(0, 0)[0]) == [clause << 1]

    def test_propagate_clauses_returns_the_clause_that_fails(self) -> None:
        solver = new_solver(2)
        clause, _ = add_clause(solver, [ge(0, 5), le(1, 3)])
        set_bound(solver, ge(1, 4), 1)
        set_bound(solver, le(0, 4), 1)
        assert propagate(solver) == clause

    def test_propagate_clauses_stops_when_the_implication_trail_is_full_and_continues(self) -> None:
        # the reserve takes the last free row: the propagation stops at the row that it visits, and visits it again
        solver = new_solver(2)
        add_clause(solver, [ge(0, 5), le(1, 3)])
        set_bound(solver, le(0, 4), 1)
        head = solver.clause_head[0]
        reserve = len(solver.itrail) - solver.itrail_top[0]
        assert propagate(solver, reserve) == CLAUSES_ITRAIL_FULL
        assert solver.clause_head[0] == head
        assert solver.domains[1].tolist() == [0, 9]
        assert propagate(solver) == CLAUSES_PROPAGATED
        assert solver.domains[1].tolist() == [0, 3]

    def test_backtrack_moves_the_clause_head_down(self) -> None:
        # the rows that a backtrack writes are new: the clause propagation must visit them, so its head goes down
        solver = LcgSolver(QueensProblem(4), log_level=LOG_LEVEL_ERROR)
        assert solver._run_search_loop()[1] is not None  # a solution, with the choice points that led to it
        assert solver.clause_head[0] == solver.itrail_top[0]
        assert solver._backtrack()
        assert solver.clause_head[0] < solver.itrail_top[0]  # the row of the parked alternative

    def test_reset_to_root_applies_the_unit_clauses(self) -> None:
        # design/proposals/lcg.md 4.7: a restart must not forget the unit clauses
        solver = new_solver(2)
        store_unit(solver.unit_lits, solver.store_top, *le(1, 3))
        solver._reset_to_root()
        assert solver.domains.tolist() == [[0, 9], [0, 3]]
        assert last_row(solver) == (le(1, 3), REASON_ROOT, -1)
        assert solver.root_failed[0] == 0

    def test_reset_to_root_finds_a_unit_clause_that_is_false_at_the_root(self) -> None:
        solver = new_solver(1)
        store_unit(solver.unit_lits, solver.store_top, *ge(0, 10))
        solver._reset_to_root()
        assert solver.root_failed[0] == 1
        assert solver._run_search_loop()[1] is None  # no solution is left

    def test_a_new_search_empties_the_store(self) -> None:
        # the clauses of an enumeration remove the solutions that it found: the next search must not keep them
        problem = AllIntervalSeriesProblem(7, True)
        solutions = [solution.tolist() for solution in BacktrackSolver(problem).find_all()]
        solver = LcgSolver(problem, log_level=LOG_LEVEL_ERROR)
        first = [solution.tolist() for solution in solver.find_all()]
        assert solver.store_top[STORE_CLAUSE_NB] + solver.store_top[STORE_UNIT_NB] > 0  # the case must learn
        first_choice_nb = solver.statistics[STATS_IDX_SOLVER_CHOICE_NB]
        second = [solution.tolist() for solution in solver.find_all()]
        assert first == second == solutions
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == 2 * first_choice_nb
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 2 * len(solutions)


def new_solver(variable_nb: int) -> LcgSolver:
    # variables in [0, 9] and no propagator: only the clauses of the test change the domains
    return LcgSolver(Problem([(0, 9)] * variable_nb), log_level=LOG_LEVEL_ERROR)


def set_bound(solver: LcgSolver, literal: tuple[int, int], choicepoint: int) -> None:
    var_side, value = literal
    solver.state[var_side] = value
    solver.itrail_top[0] = itrail_push(
        solver.itrail,
        solver.itrail_last,
        solver.itrail_top[0],
        var_side,
        value,
        choicepoint,
        REASON_DECISION,
        -1,
        -1,
        -1,
    )


def add_clause(solver: LcgSolver, literals: list[tuple[int, int]]) -> tuple[int, bool]:
    clause = store_clause(
        solver.clause_lits, solver.clause_start, solver.store_top, np.array(literals, dtype=np.int32), len(literals)
    )
    attached = attach_clause(
        clause,
        solver.clause_lits,
        solver.clause_start,
        solver.watch_head,
        solver.watches,
        solver.state,
        solver.utrail_log,
        solver.utrail_top,
        solver.utrail_indices,
        0,
        solver.itrail,
        solver.itrail_top,
        solver.itrail_last,
        solver.choicepoint_top[0],
        solver.triggered_propagators,
        solver.entailed,
        solver.problem.triggers,
        solver.problem.triggers_offsets,
        solver.problem.propagator_infos,
    )
    return clause, attached


def propagate(solver: LcgSolver, itrail_reserve: int = 0) -> int:
    status, utrail_sz, itrail_sz = propagate_clauses(
        solver.state,
        solver.utrail_log,
        solver.utrail_indices,
        0,
        solver.utrail_top[0],
        solver.itrail,
        solver.itrail_last,
        solver.itrail_top[0],
        itrail_reserve,
        1,
        solver.clause_head,
        solver.clause_lits,
        solver.clause_start,
        solver.watch_head,
        solver.watches,
        solver.triggered_propagators,
        solver.entailed,
        solver.problem.triggers,
        solver.problem.triggers_offsets,
        solver.problem.propagator_infos,
    )
    solver.utrail_top[0] = utrail_sz
    solver.itrail_top[0] = itrail_sz
    return status


def watched(solver: LcgSolver, clause: int) -> list[tuple[int, int]]:
    start = solver.clause_start[clause]
    return [(var_side, value) for var_side, value in solver.clause_lits[start : start + 2].tolist()]


def watches_of(solver: LcgSolver, bound: int) -> list[int]:
    watches = []
    watch = solver.watch_head[bound]
    while watch >= 0:
        watches.append(int(watch))
        watch = solver.watches[watch, WATCH_NEXT]
    return watches


def last_row(solver: LcgSolver) -> tuple[tuple[int, int], int, int]:
    row = solver.itrail[solver.itrail_top[0] - 1]
    return (
        (int(row[ITRAIL_VAR_SIDE]), int(row[ITRAIL_VALUE])),
        int(row[ITRAIL_REASON_KIND]),
        int(row[ITRAIL_REASON_DATA]),
    )
