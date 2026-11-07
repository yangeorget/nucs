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
from functools import partial

import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.examples.golomb.golomb_problem import GolombProblem
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_DOM_WDEG, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.implication_trail import REASON_DECISION, REASON_PROPAGATOR
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.solvers.restart_policies import RESTART_LUBY, RestartPolicy
from nucs.solvers.searches import Search
from nucs.solvers.solver import OPTIM_PRUNE, OPTIM_RESET
from nucs.solvers.tree_search_solver import SOLVER_RUNNING, TreeSearchSolver
from nucs.statistics import STATS_IDX_SOLVER_ELAPSED_TIME, STATS_IDX_SOLVER_RESTART_NB


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
    def test_same_counters_as_backtrack_solver(
        self, run: Callable[[Callable[..., TreeSearchSolver]], tuple[TreeSearchSolver, list]]
    ) -> None:
        # steps 4 and 5 of design/proposals/lcg.md 7.0: LcgSolver searches as BacktrackSolver does, so every counter,
        # the weights of the propagators and the solutions are the same; only the elapsed time can differ. The debug
        # mode replays the implication trail at each node: restarts, the objective at the root and the pruning of
        # OPTIM_PRUNE all write it.
        backtrack_solver, backtrack_solutions = run(BacktrackSolver)
        lcg_solver, lcg_solutions = run(partial(LcgSolver, debug=True))
        assert lcg_solutions == backtrack_solutions
        backtrack_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        lcg_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        assert lcg_solver.statistics.tolist() == backtrack_solver.statistics.tolist()
        assert lcg_solver.propagator_weights.tolist() == backtrack_solver.propagator_weights.tolist()

    def test_run_search_loop_records_the_decisions_and_the_propagations(self) -> None:
        problem = Problem([(0, 1), (0, 1)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(2))
        solver = LcgSolver(problem, debug=True)
        status, solution = solver._run_search_loop()
        assert status == SOLVER_RUNNING
        assert solution is not None
        assert solution.tolist() == [0, 1]
        # the decision [x0 <= 0] opens level 1; alldifferent, the propagator 0, then sets [x1 >= 1] in a call that
        # starts at row 1
        assert solver.itrail_top.tolist() == [2]
        assert solver.itrail[:2].tolist() == [
            [0 << 1 | DOMAIN_MAX, 0, -1, 1, REASON_DECISION, -1, -1],
            [1 << 1 | DOMAIN_MIN, 1, -1, 1, REASON_PROPAGATOR, 0, 1],
        ]
        assert solver.level_start[1] == 0
        assert solver._backtrack()
        # back at level 0: the rows of level 1 are gone, and the parked alternative [x0 >= 1] is the new row
        assert solver.itrail_top.tolist() == [1]
        assert solver.itrail[0].tolist() == [0 << 1 | DOMAIN_MIN, 1, -1, 0, REASON_DECISION, -1, -1]
        assert solver.itrail_last.tolist() == [0, -1, -1, -1]

    @pytest.mark.parametrize("row_nb", [1, 13, 14])  # below and at 14, the room of a call on 6 queens: 2 x 6 + 2
    def test_find_all_the_implication_trail_grows_rather_than_overruns(self, row_nb: int) -> None:
        # a propagation that stops for room continues where it stopped: the counters are the same as without growth
        backtrack_solver = BacktrackSolver(QueensProblem(6))
        backtrack_solutions = [solution.tolist() for solution in backtrack_solver.find_all()]
        solver = LcgSolver(QueensProblem(6), debug=True)
        solver.itrail = solver.itrail[:row_nb].copy()
        assert [solution.tolist() for solution in solver.find_all()] == backtrack_solutions
        assert len(solver.itrail) > row_nb  # it did have to grow
        backtrack_solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        solver.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] = 0
        assert solver.statistics.tolist() == backtrack_solver.statistics.tolist()

    def test_find_all_level_start_grows_with_the_stack(self) -> None:
        backtrack_solutions = [solution.tolist() for solution in BacktrackSolver(QueensProblem(7)).find_all()]
        solver = LcgSolver(QueensProblem(7), debug=True)
        solver.choice_point_stk = solver.choice_point_stk[:4].copy()
        solver.level_start = solver.level_start[:4].copy()
        assert [solution.tolist() for solution in solver.find_all()] == backtrack_solutions
        assert len(solver.level_start) == len(solver.choice_point_stk) > 4

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
