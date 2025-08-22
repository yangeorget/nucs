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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import pytest

from nucs.constants import OPTIM_PRUNE, OPTIM_RESET, STATS_LBL_SOLUTION_NB, STATS_LBL_SOLVER_CHOICE_DEPTH
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    DOM_HEURISTIC_SPLIT_LOW,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_LEQ, ALG_ALLDIFFERENT, ALG_RELATION
from nucs.solvers.backtrack_solver import BacktrackSolver, get_function_addresses, solve_one
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
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        solution = solve_one(
            problem.propagator_nb,
            solver.statistics,
            problem.algorithms,
            problem.bounds,
            problem.props_variables,
            problem.props_parameters,
            problem.triggers,
            solver.domains_stk,
            solver.not_entailed_propagators_stk,
            solver.dom_update_stk,
            solver.stks_top,
            solver.triggered_propagators,
            solver.consistency_alg_idx,
            solver.decision_variables,
            solver.var_heuristic_idx,
            solver.var_heuristic_params,
            solver.dom_heuristic_idx,
            solver.dom_heuristic_params,
            compute_domains_addrs,
            consistency_alg_addrs,
            var_heuristic_addrs,
            dom_heuristic_addrs,
        )
        assert solution.tolist() == [0, 0]
        assert solver.stks_top == 2
        assert solver.domains_stk[0, 0].tolist() == [1, 1]
        assert solver.domains_stk[0, 1].tolist() == [0, 1]
        assert solver.domains_stk[1, 0].tolist() == [0, 0]
        assert solver.domains_stk[1, 1].tolist() == [1, 1]
        assert backtrack(
            problem.propagator_nb,
            solver.statistics,
            solver.not_entailed_propagators_stk,
            solver.dom_update_stk,
            solver.stks_top,
            solver.triggered_propagators,
            problem.triggers,
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

    def test_minimize_affine_leq(self) -> None:
        problem = Problem([(2, 5), (2, 5), (0, 10)])
        problem.add_propagator(ALG_AFFINE_LEQ, [0, 1, 2], [1, 1, -1, 0])
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
        solver = BacktrackSolver(problem, dom_heuristic_idx=dom_heuristic)
        solution = solver.maximize(0, mode=mode)
        assert solution is not None
        assert solution.tolist() == [5]
        statistics = solver.get_statistics_as_dictionary()
        assert statistics[STATS_LBL_SOLUTION_NB] == solution_nb
