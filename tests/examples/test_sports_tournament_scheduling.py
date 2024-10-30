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
# Copyright 2024 - Yan Georget
###############################################################################
from nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem import (
    SportsTournamentSchedulingProblem,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestSportsTournamentScheduling:
    def test_teams_per_week(self) -> None:
        problem = SportsTournamentSchedulingProblem(4)
        assert problem.teams_per_week(0) == [0, 1, 6, 7]
        assert problem.teams_per_week(1) == [2, 3, 8, 9]
        assert problem.teams_per_week(2) == [4, 5, 10, 11]

    def test_teams_per_period(self) -> None:
        problem = SportsTournamentSchedulingProblem(4)
        assert problem.teams_per_period(0) == [0, 1, 2, 3, 4, 5]
        assert problem.teams_per_period(1) == [6, 7, 8, 9, 10, 11]

    def test_plays(self) -> None:
        problem = SportsTournamentSchedulingProblem(4)
        assert problem.plays() == [0, 1, 0, 0, 2, 1, 0, 3, 2, 1, 2, 3, 1, 3, 4, 2, 3, 5]

    def test_sports_tournament_scheduling_check(self) -> None:
        problem = SportsTournamentSchedulingProblem(8, False)
        problem.shr_domains_lst[:56] = [
            [v, v]
            for v in [
                0,
                1,
                0,
                2,
                4,
                7,
                3,
                6,
                3,
                7,
                1,
                5,
                2,
                4,
                2,
                3,
                1,
                7,
                0,
                3,
                5,
                7,
                1,
                4,
                0,
                6,
                5,
                6,
                4,
                5,
                3,
                5,
                1,
                6,
                0,
                4,
                2,
                6,
                2,
                7,
                0,
                7,
                6,
                7,
                4,
                6,
                2,
                5,
                1,
                2,
                0,
                5,
                3,
                4,
                1,
                3,
            ]
        ]
        solver = BacktrackSolver(problem)
        assert next(solver.solve()) is not None

    def test_sports_tournament_scheduling_solve(self) -> None:
        problem = SportsTournamentSchedulingProblem(8)
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE
        )
        assert next(solver.solve()) is not None
