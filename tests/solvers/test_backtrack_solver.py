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
import time

import pytest

from nucs.buckets import buckets_empty
from nucs.constants import (
    LEVEL_BOUND,
    LEVEL_VALUE,
    LEVEL_VARIABLE,
    MAX,
    MIN,
    OPTIM_PRUNE,
    OPTIM_RESET,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_GREATEST_DOMAIN,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_ALLDIFFERENT,
    ALG_LINEAR_LEQ_C,
    ALG_LINEAR_NEQ_C,
    ALG_NEQ,
    ALG_RELATION,
    IDEMPOTENT,
)
from nucs.solvers.backtrack_solver import BacktrackSolver, solve_one
from nucs.solvers.choice_points import backtrack
from nucs.solvers.search import Search


class TestBacktrackSolver:
    def test_algorithm_statistics_partition_the_global_counters(self) -> None:
        """The per-algorithm counters are a breakdown of the global ones, so they must sum to them.

        This is what makes them usable for a throughput investigation: a no-change share attributed to one
        algorithm is a share of the whole, not of some separately-counted subset.
        """
        problem = Problem([(0, 5)] * 4)
        problem.add_propagator(ALG_ALLDIFFERENT, range(4))
        problem.add_propagator(ALG_LINEAR_LEQ_C, range(4), [1, 1, 1, 1, 12])
        problem.add_propagator(ALG_NEQ, [0, 3])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        statistics = solver.get_statistics_as_dictionary()
        names = {"ALLDIFFERENT", "LINEAR_LEQ_C", "NEQ"}
        calls = {name: statistics[f"{STATS_LBL_PROPAGATOR_FILTER_NB}_{name}"] for name in names}
        no_changes = {name: statistics[f"{STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB}_{name}"] for name in names}
        assert sum(calls.values()) == statistics[STATS_LBL_PROPAGATOR_FILTER_NB]
        assert sum(no_changes.values()) == statistics[STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB]
        # a propagator cannot change nothing more often than it was called
        assert all(no_changes[name] <= calls[name] for name in names)

    def test_algorithm_statistics_omit_algorithms_that_never_ran(self) -> None:
        """Only the algorithms a problem actually uses appear, so the breakdown stays readable."""
        problem = Problem([(0, 3)] * 2)
        problem.add_propagator(ALG_NEQ, [0, 1])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        statistics = solver.get_statistics_as_dictionary()
        assert [key for key in statistics if key.startswith(f"{STATS_LBL_PROPAGATOR_FILTER_NB}_")] == [
            f"{STATS_LBL_PROPAGATOR_FILTER_NB}_NEQ"
        ]

    def test_solve_stops_at_the_timeout(self) -> None:
        """A timeout cuts the enumeration short and says so, instead of silently looking exhausted."""
        problem = Problem([(0, 299), (0, 299)])
        solver = BacktrackSolver(problem)
        solutions = sum(1 for _ in solver.solve(timeout=0.05))
        assert solver.timed_out
        assert 0 < solutions < 90000

    def test_solve_without_timeout_is_exhaustive(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = BacktrackSolver(problem)
        assert sum(1 for _ in solver.solve()) == 10000
        assert not solver.timed_out

    def test_find_best_returns_the_best_found_within_the_timeout(self) -> None:
        """Under a timeout find_best still returns a solution -- just not a proven optimum."""
        problem = Problem([(1, 500)])
        solver = BacktrackSolver(problem)
        # each improving solution costs the consumer 20 ms, so the budget runs out well before 500
        best = None
        for best in solver.optimize(0, MAX, OPTIM_RESET, timeout=0.10):
            time.sleep(0.02)
        assert solver.timed_out
        assert best is not None
        assert best[0] < 500

    def test_solve_all(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 10000
        assert statistics[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_solve_one(self) -> None:
        """The state a solution leaves behind, and what backtracking restores of it.

        This used to assert the opposite property: that every level kept its own snapshot and that
        backtracking left them all untouched, because the pointer decrement was the whole of the restore.
        There is one set of domains now, and backtracking restores it by replaying the undo log.
        """
        problem = Problem([(0, 1), (0, 1)])
        solver = BacktrackSolver(problem)
        buckets_empty(solver.triggered_propagators, problem.priorities)
        solution = solve_one(
            problem.propagator_nb,
            solver.statistics,
            problem.algorithms,
            problem.priorities,
            problem.offsets,
            problem.propagator_variables,
            problem.propagator_parameters,
            problem.triggers,
            problem.triggers_offsets,
            solver.state,
            solver.domains,
            solver.entailed,
            solver.trail,
            solver.trail_top,
            solver.pos,
            solver.level_stk,
            solver.stks_top,
            solver.triggered_propagators,
            solver.consistency_alg_fcts,
            solver.decision_variables,
            solver.decision_variables_offsets,
            solver.var_heuristic_fcts,
            solver.var_heuristic_params,
            solver.var_heuristic_params_offsets,
            solver.var_heuristic_params_shapes,
            solver.dom_heuristic_fcts,
            solver.dom_heuristic_params,
            solver.dom_heuristic_params_offsets,
            solver.dom_heuristic_params_shapes,
            solver.compute_domains_fcts,
            solver.domain_buffer,
            IDEMPOTENT,
            solver.objective,
            solver.decision,
            solver.status,
            solver.trail_headroom,
        )
        assert solution is not None
        assert solution.tolist() == [0, 0]
        assert solver.stks_top == 2
        # two min_value decisions, so both variables are ground at [0, 0]
        assert solver.domains[0].tolist() == [0, 0]
        assert solver.domains[1].tolist() == [0, 0]
        # each level parked the refutation of its own decision: raise that variable's min to 1
        assert solver.level_stk[0, LEVEL_VARIABLE] == 0
        assert (solver.level_stk[0, LEVEL_BOUND], solver.level_stk[0, LEVEL_VALUE]) == (MIN, 1)
        assert solver.level_stk[1, LEVEL_VARIABLE] == 1
        assert (solver.level_stk[1, LEVEL_BOUND], solver.level_stk[1, LEVEL_VALUE]) == (MIN, 1)
        assert backtrack(
            solver.statistics,
            solver.state,
            solver.trail,
            solver.trail_top,
            solver.pos,
            solver.level_stk,
            solver.stks_top,
            solver.entailed,
            solver.triggered_propagators,
            problem.triggers,
            problem.triggers_offsets,
            problem.priorities,
            problem.propagator_nb,
            solver.objective,
        )
        # back at level 1, with variable 1's refutation applied: variable 0 stays at its decision
        assert solver.stks_top == 1
        assert solver.domains[0].tolist() == [0, 0]
        assert solver.domains[1].tolist() == [1, 1]

    def test_the_trail_holds_only_what_the_live_branch_changed(self) -> None:
        """The trail is bounded by the changes on the path, not by domain_nb per node.

        That is the whole claim of trailing over copying, and it is checkable: a live trail of a few
        dozen entries where the copying representation would have written 30 int32 per node.
        """
        problem = Problem([(0, 9)] * 3)
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)
        solutions = solver.find_all()
        assert len(solutions) == 1000
        assert solutions[0].tolist() == [0, 0, 0]
        assert solutions[-1].tolist() == [9, 9, 9]
        assert solver.stks_top[0] == 0  # exhausted, back at the root
        # the root's own refutations are never undone -- nothing pops past level 0 -- but everything
        # deeper is, so what is left is a handful of entries rather than one snapshot per level
        assert solver.trail_top[0] < 4 * len(solver.pos)
        assert len(solver.trail) == 1 << 16  # it never had to grow

    @pytest.mark.parametrize("trail_max_size", [8, 9, 16, 17, 33])
    def test_the_trail_grows_rather_than_overruns(self, trail_max_size: int) -> None:
        """A trail too small for the search is grown, not overrun -- and the search is not restarted.

        Parameterized over sizes below and just above the headroom the solver reserves, so that a
        headroom too small for one step of the search shows up as a wrong answer rather than as luck.
        """

        def build() -> Problem:
            problem = Problem([(0, 7), (0, 7), (0, 7)])
            problem.add_propagator(ALG_ALLDIFFERENT, range(3))
            return problem

        reference = [solution.tolist() for solution in BacktrackSolver(build()).find_all()]
        assert len(reference) == 336
        solver = BacktrackSolver(build(), trail_max_size=trail_max_size)
        assert [solution.tolist() for solution in solver.find_all()] == reference
        assert len(solver.trail) > trail_max_size  # it did have to grow

    def test_the_level_stack_grows_rather_than_overruns(self) -> None:
        """Likewise for a search deeper than the level stack: grow, do not corrupt memory."""
        problem = Problem([(0, 5)] * 6)
        reference = BacktrackSolver(problem).find_all()
        solver = BacktrackSolver(Problem([(0, 5)] * 6), stks_max_height=4)
        assert len(solver.find_all()) == len(reference)
        assert len(solver.level_stk) > 4

    def test_find_all(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        solver = BacktrackSolver(problem)
        solutions = solver.find_all()
        assert len(solutions) == 4
        assert solutions[0].tolist() == [0, 0]
        assert solutions[1].tolist() == [0, 1]
        assert solutions[2].tolist() == [1, 0]
        assert solutions[3].tolist() == [1, 1]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 4
        assert statistics[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_find_all_alldifferent(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagator(ALG_ALLDIFFERENT, [0, 1, 2])
        solver = BacktrackSolver(problem)
        solutions = solver.find_all()
        assert len(solutions) == 6
        assert solutions[0].tolist() == [0, 1, 2]
        assert solutions[1].tolist() == [0, 2, 1]
        assert solutions[2].tolist() == [1, 0, 2]
        assert solutions[3].tolist() == [1, 2, 0]
        assert solutions[4].tolist() == [2, 0, 1]
        assert solutions[5].tolist() == [2, 1, 0]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 6

    def test_sequential_search(self) -> None:
        # two searches: the first branches variable 0 (indomain_max), the second variable 1 (indomain_min)
        problem = Problem([(1, 3), (1, 3)])
        problem.add_propagator(ALG_NEQ, [0, 1])  # x != y
        solver = BacktrackSolver(
            problem,
            searches=[
                Search([0], VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, [[]], DOM_HEURISTIC_MAX_VALUE, [[]]),
                Search([1], VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, [[]], DOM_HEURISTIC_MIN_VALUE, [[]]),
            ],
        )
        solutions = solver.find_all()
        # variable 0 takes its largest value first, then variable 1 its smallest (the only constraint is x != y)
        assert solutions[0].tolist() == [3, 1]
        # the sequential search still enumerates every solution, just in a different order
        assert len(solutions) == 6
        assert all(x != y for x, y in (s.tolist() for s in solutions))

    def test_sequential_search_second_group_only_after_first_bound(self) -> None:
        # the second search must stay dormant until every decision variable of the first search is bound:
        # variable 2 (searched first) is fixed to its max before variables 0 and 1 are touched
        problem = Problem([(0, 9), (0, 9), (0, 9)])
        solver = BacktrackSolver(
            problem,
            searches=[
                Search([2], VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, [[]], DOM_HEURISTIC_MAX_VALUE, [[]]),
                Search([0, 1], VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, [[]], DOM_HEURISTIC_MIN_VALUE, [[]]),
            ],
        )
        solution = next(solver.solve())
        assert solution.tolist() == [0, 0, 9]  # variable 2 grounded to 9 first, then 0 and 1 to their min

    def test_split_grounding_wakes_ground_triggered_propagator(self) -> None:
        # A split heuristic that grounds a variable in its current branch must report a GROUND event,
        # otherwise a propagator woken only on ground events (here linear_neq_c) never fires and an
        # inconsistent solution slips through. Regression for indomain_reverse_split + anti_first_fail.
        for dom_heuristic in (DOM_HEURISTIC_SPLIT_HIGH, DOM_HEURISTIC_SPLIT_LOW):
            problem = Problem([(1, 5), (4, 5)])
            problem.add_propagator(ALG_LINEAR_NEQ_C, [0, 1], [1, -1, 0])  # x != y
            solver = BacktrackSolver(problem, var_heuristic=VAR_HEURISTIC_GREATEST_DOMAIN, dom_heuristic=dom_heuristic)
            solutions = solver.find_all()
            assert len(solutions) == 8
            assert all(x != y for x, y in (s.tolist() for s in solutions))

    def test_minimize_relation(self) -> None:
        problem = Problem([(-5, 5), (-100, 100)])
        problem.add_propagator(
            ALG_RELATION, [0, 1], [-5, 25, -4, 16, -3, 9, -2, 4, -1, 1, 0, 0, 1, 1, 2, 4, 3, 9, 4, 16, 5, 25]
        )
        solver = BacktrackSolver(problem)
        solution = solver.find_best(1, bound=MIN)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 6

    def test_minimize_linear_leq_c(self) -> None:
        problem = Problem([(2, 5), (2, 5), (0, 10)])
        problem.add_propagator(ALG_LINEAR_LEQ_C, [0, 1, 2], [1, 1, -1, 0])
        solver = BacktrackSolver(problem)
        solution = solver.find_best(2, bound=MIN)
        assert solution is not None
        assert solution.tolist() == [2, 2, 4]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 1

    @pytest.mark.parametrize(
        "mode,dom_heuristic, solution_nb",
        [
            (OPTIM_PRUNE, DOM_HEURISTIC_MIN_VALUE, 5),
            (OPTIM_PRUNE, DOM_HEURISTIC_MID_VALUE, 3),
            (OPTIM_PRUNE, DOM_HEURISTIC_SPLIT_LOW, 5),
            (OPTIM_PRUNE, DOM_HEURISTIC_SPLIT_HIGH, 1),
            (OPTIM_RESET, DOM_HEURISTIC_MIN_VALUE, 5),
            (OPTIM_RESET, DOM_HEURISTIC_MID_VALUE, 3),
            (OPTIM_RESET, DOM_HEURISTIC_SPLIT_LOW, 5),
            (OPTIM_RESET, DOM_HEURISTIC_SPLIT_HIGH, 1),
        ],
    )
    def test_maximize(self, mode: str, dom_heuristic: int, solution_nb: int) -> None:
        problem = Problem([(1, 5)])
        solver = BacktrackSolver(problem, dom_heuristic=dom_heuristic)
        solution = solver.find_best(0, bound=MAX, mode=mode)
        assert solution is not None
        assert solution.tolist() == [5]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == solution_nb

    @pytest.mark.parametrize(
        "dom_heuristic",
        [
            DOM_HEURISTIC_MIN_VALUE,
            DOM_HEURISTIC_MAX_VALUE,
            DOM_HEURISTIC_MID_VALUE,
            DOM_HEURISTIC_SPLIT_LOW,
            DOM_HEURISTIC_SPLIT_HIGH,
        ],
    )
    @pytest.mark.parametrize("bound", [MIN, MAX])
    def test_prune_and_reset_yield_the_same_optimization_sequence(self, dom_heuristic: int, bound: int) -> None:
        """OPTIM_PRUNE resumes the search where it was instead of restarting, but the improving solutions
        it reports must be the ones OPTIM_RESET reports -- pruning is an optimization, not a semantics.

        Regression test for a hang: OPTIM_PRUNE used to rewrite the tightened bound into every stored
        choice point and drop one level per wipe-out, which assumes the wiped levels are the deepest ones.
        A three-way split (DOM_HEURISTIC_MID_VALUE, a DECISION_EQ) makes two levels siblings
        holding disjoint objective ranges, so minimizing wiped the shallower one and the count-based drop
        discarded the survivor. The resulting level had an empty domain that no variable heuristic could
        claim and no propagator noticed, and solve_one span forever on an empty queue.
        Only maximization was covered before, which wipes the deeper level first and so never hit it.
        """
        expected = [
            solution.tolist()
            for solution in BacktrackSolver(Problem([(1, 5)]), dom_heuristic=dom_heuristic, log_level="ERROR").optimize(
                0, bound, OPTIM_RESET
            )
        ]
        actual = [
            solution.tolist()
            for solution in BacktrackSolver(Problem([(1, 5)]), dom_heuristic=dom_heuristic, log_level="ERROR").optimize(
                0, bound, OPTIM_PRUNE
            )
        ]
        assert actual == expected

    def test_prune_terminates_on_a_three_way_split(self) -> None:
        """The minimal reproduction of the hang above, pinned to its exact expected output."""
        solver = BacktrackSolver(Problem([(1, 5)]), dom_heuristic=DOM_HEURISTIC_MID_VALUE, log_level="ERROR")
        assert [solution.tolist() for solution in solver.optimize(0, MIN, OPTIM_PRUNE)] == [[3], [1]]
