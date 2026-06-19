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
import pytest

from nucs.buckets import buckets_empty, STORAGE_OFFSET
from nucs.constants import (
    OPTIM_PRUNE,
    OPTIM_RESET,
    STATS_LBL_ALG_SHAVING_NB,
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
    ALG_LINEAR_LEQ_C,
    ALG_LINEAR_NEQ_C,
    ALG_ALLDIFFERENT,
    ALG_NEQ,
    ALG_RELATION,
    get_algorithm_nb,
)
from nucs.solvers.backtrack_solver import BacktrackSolver, Search, solve_one
from nucs.solvers.choice_points import backtrack
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_SHAVING


class TestBacktrackSolver:
    def test_solve_all(self) -> None:
        problem = Problem([(0, 99), (0, 99)])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 10000
        assert statistics[STATS_LBL_SOLVER_CHOICE_DEPTH] == 2

    def test_solve_one(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        solver = BacktrackSolver(problem, stks_max_height=3)
        buckets_empty(solver.triggered_propagators, problem.priorities)
        solution = solve_one(
            get_algorithm_nb(),
            problem.propagator_nb,
            solver.statistics,
            problem.algorithms,
            problem.priorities,
            problem.bounds,
            problem.propagator_variables,
            problem.propagator_parameters,
            problem.triggers,
            problem.triggers_offsets,
            solver.domains_stk,
            solver.entailed_propagator_depths,
            solver.entailment_trail,
            solver.domain_update_stk,
            solver.unbound_variable_nb_stk,
            solver.stks_top,
            solver.triggered_propagators,
            solver.consistency_alg_fcts,
            solver.decision_variables,
            solver.var_heuristic_fcts,
            solver.var_heuristic_params,
            solver.dom_heuristic_fcts,
            solver.dom_heuristic_params,
            solver.compute_domains_fcts,
            solver.domain_buffer,
        )
        assert solution is not None
        assert solution.tolist() == [0, 0]
        assert solver.stks_top == 2
        assert solver.domains_stk[0, 0].tolist() == [1, 1]
        assert solver.domains_stk[0, 1].tolist() == [0, 1]
        assert solver.domains_stk[1, 0].tolist() == [0, 0]
        assert solver.domains_stk[1, 1].tolist() == [1, 1]
        membership_offset = STORAGE_OFFSET + problem.propagator_nb
        assert backtrack(
            solver.statistics,
            solver.entailed_propagator_depths,
            solver.entailment_trail,
            solver.domain_update_stk,
            solver.stks_top,
            solver.triggered_propagators,
            problem.triggers,
            problem.triggers_offsets,
            problem.priorities,
            membership_offset,
        )
        assert solver.stks_top == 1
        assert solver.domains_stk[0, 0].tolist() == [1, 1]
        assert solver.domains_stk[0, 1].tolist() == [0, 1]
        assert solver.domains_stk[1, 0].tolist() == [0, 0]
        assert solver.domains_stk[1, 1].tolist() == [1, 1]

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

    def test_shaving_consistency_algorithm_matches_bound_consistency(self) -> None:
        # the shaving consistency algorithm must reach exactly the same solutions as plain bound consistency.
        # Shaving a size-2 domain used to leak the unbound-variable count (the leftover sub-domain grounds,
        # decrementing the count, which the failed-shave restore did not undo), so the count underflowed and
        # the solver looped forever; this test both checks correctness and guards against that regression.
        def solve_with(consistency_algorithm: int) -> "tuple":
            problem = Problem([(0, 4), (0, 4), (0, 4), (0, 4)])
            problem.add_propagator(ALG_ALLDIFFERENT, [0, 1, 2, 3])
            problem.add_propagator(ALG_LINEAR_LEQ_C, [0, 1, 2, 3], [1, 1, 1, 1, 6])  # sum <= 6
            solver = BacktrackSolver(problem, consistency_algorithm=consistency_algorithm)
            return sorted(tuple(s.tolist()) for s in solver.solve()), solver

        bc_solutions, _ = solve_with(CONSISTENCY_ALG_BC)
        shaving_solutions, shaving_solver = solve_with(CONSISTENCY_ALG_SHAVING)
        assert len(bc_solutions) == 24
        assert shaving_solutions == bc_solutions
        # the shaving branch was genuinely exercised (not silently skipped)
        assert shaving_solver.get_statistics_as_dictionary()[STATS_LBL_ALG_SHAVING_NB] > 0

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
        solution = solver.minimize(1)
        assert solution is not None
        assert solution.tolist() == [0, 0]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == 6

    def test_minimize_linear_leq_c(self) -> None:
        problem = Problem([(2, 5), (2, 5), (0, 10)])
        problem.add_propagator(ALG_LINEAR_LEQ_C, [0, 1, 2], [1, 1, -1, 0])
        solver = BacktrackSolver(problem)
        solution = solver.minimize(2)
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
        solution = solver.maximize(0, mode=mode)
        assert solution is not None
        assert solution.tolist() == [5]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == solution_nb
