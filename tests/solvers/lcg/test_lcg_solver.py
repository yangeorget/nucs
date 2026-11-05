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
from collections.abc import Callable

import pytest

from nucs.constants import DOMAIN_MIN
from nucs.examples.golomb.golomb_problem import GolombProblem
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_DOM_WDEG, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.solvers.restart_policies import RESTART_LUBY, RestartPolicy
from nucs.solvers.searches import Search
from nucs.solvers.solver import OPTIM_PRUNE, OPTIM_RESET
from nucs.solvers.tree_search_solver import TreeSearchSolver
from nucs.statistics import STATS_IDX_SOLVER_ELAPSED_TIME, STATS_IDX_SOLVER_RESTART_NB


def run_queens_find_all(solver_class: type[TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    solver = solver_class(QueensProblem(7))
    return solver, [solution.tolist() for solution in solver.find_all()]


def run_queens_two_searches(solver_class: type[TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
    # a sequential search: the first search branches on the first half of the queens, the second on the rest
    searches = [
        Search(decision_variables=range(4), var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN),
        Search(decision_variables=range(4, 8), dom_heuristic=DOM_HEURISTIC_SPLIT_LOW),
    ]
    solver = solver_class(QueensProblem(8), searches=searches)
    return solver, [solution.tolist() for solution in solver.find_all()]


def run_golomb_find_best(solver_class: type[TreeSearchSolver], mode: str) -> tuple[TreeSearchSolver, list]:
    problem = GolombProblem(6)
    solver = solver_class(problem)
    solution = solver.find_best(problem.length_idx, DOMAIN_MIN, mode=mode)
    return solver, [None if solution is None else solution.tolist()]


def run_quasigroup_restarts(solver_class: type[TreeSearchSolver]) -> tuple[TreeSearchSolver, list]:
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
    def test_same_counters_as_backtrack_solver(
        self, run: Callable[[type[TreeSearchSolver]], tuple[TreeSearchSolver, list]]
    ) -> None:
        # step 4 of design/proposals/lcg.md 7.0: LcgSolver searches as BacktrackSolver does, so every counter, the
        # weights of the propagators and the solutions are the same; only the elapsed time can differ
        backtrack_solver, backtrack_solutions = run(BacktrackSolver)
        lcg_solver, lcg_solutions = run(LcgSolver)
        assert lcg_solutions == backtrack_solutions
        backtrack_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        lcg_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        assert lcg_solver.statistics.tolist() == backtrack_solver.statistics.tolist()
        assert lcg_solver.propagator_weights.tolist() == backtrack_solver.propagator_weights.tolist()
