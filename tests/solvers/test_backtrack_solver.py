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
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_GREATEST_DOMAIN,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_LINEAR_LEQ_C,
    ALG_LINEAR_NEQ_C,
    ALG_ALLDIFFERENT,
    ALG_RELATION,
    get_algorithm_nb,
)
from nucs.solvers.backtrack_solver import BacktrackSolver, solve_one
from nucs.solvers.choice_points import backtrack


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
