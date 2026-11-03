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
from typing import Any

import pytest

from nucs.examples.default_argument_parser import (
    DefaultArgumentParser,
    run_solver,
    solver_kwargs_from_args,
    timeout_from_args,
)
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC
from nucs.solvers.restart_policies import RESTART_LUBY, RestartPolicy
from nucs.solvers.searches import Search
from nucs.statistics import STATS_IDX_SOLUTION_NB


class TestDefaultArgumentParser:
    @pytest.mark.parametrize(
        "args, expected_args",
        [
            (["--consistency-algorithm", "BC"], {"consistency_algorithm": "BC"}),
            (["--display-solutions"], {"display_solutions": True}),
            (["--display-stats"], {"display_stats": True}),
            (["--dom-heuristic", "MIN_VALUE"], {"dom_heuristic": "MIN_VALUE"}),
            (["--find-all"], {"find_all": True}),
            ([], {"last_conflict": None}),
            (["--last-conflict"], {"last_conflict": True}),
            (["--no-last-conflict"], {"last_conflict": False}),
            (["--log-level", "INFO"], {"log_level": "INFO"}),
            (["--optimization-mode", "RESET"], {"optimization_mode": "RESET"}),
            ([], {"restart": None, "time_limit": None}),
            (["--restart", "luby,100"], {"restart": RestartPolicy(RESTART_LUBY, 100)}),
            (["--time-limit", "500"], {"time_limit": 500}),
            (["--symmetry-breaking"], {"symmetry_breaking": True}),
            (["--var-heuristic", "SMALLEST_DOMAIN"], {"var_heuristic": "SMALLEST_DOMAIN"}),
        ],
    )
    def test_parse_args(self, args: str, expected_args: dict[str, Any]) -> None:
        parser = DefaultArgumentParser()
        parsed_args = parser.parse_args(args)
        for arg, value in expected_args.items():
            assert getattr(parsed_args, arg) == value

    def test_solver_kwargs_from_args(self) -> None:
        parser = DefaultArgumentParser()
        args = parser.parse_args(
            ["--consistency-algorithm", "BC", "--var-heuristic", "SMALLEST_DOMAIN", "--dom-heuristic", "MIN_VALUE"]
        )
        kwargs = solver_kwargs_from_args(args, searches=[Search(dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)])
        assert kwargs["consistency_algorithm"] == CONSISTENCY_ALG_BC
        # the CLI overrides the programmatic default
        assert kwargs["searches"] == [
            Search(var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic=DOM_HEURISTIC_MIN_VALUE)
        ]

    def test_solver_kwargs_from_args_overrides_the_heuristics_of_every_search(self) -> None:
        """A heuristic given on the command line reaches each search, which keeps its parameters."""
        parser = DefaultArgumentParser()
        args = parser.parse_args(["--var-heuristic", "SMALLEST_DOMAIN"])
        searches = [Search([0], var_heuristic_params=[[1]]), Search([1], dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)]
        kwargs = solver_kwargs_from_args(args, searches=searches)
        assert kwargs["searches"] == [
            Search([0], VAR_HEURISTIC_SMALLEST_DOMAIN, [[1]]),
            Search([1], VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW),
        ]

    @pytest.mark.parametrize(
        "args, default, expected",
        [
            ([], True, True),  # no option: the example's default stays
            (["--no-last-conflict"], True, False),  # False must override, not be dropped as a missing value
            (["--last-conflict"], False, True),
        ],
    )
    def test_solver_kwargs_from_args_last_conflict(self, args: list[str], default: bool, expected: bool) -> None:
        parser = DefaultArgumentParser()
        kwargs = solver_kwargs_from_args(parser.parse_args(args), last_conflict=default)
        assert kwargs["last_conflict"] is expected

    @pytest.mark.parametrize("text", ["fibonacci", "luby,0", "luby,10,2,3"])
    def test_parse_args_rejects_a_wrong_restart(self, text: str) -> None:
        with pytest.raises(SystemExit):  # argparse reports a usage error
            DefaultArgumentParser().parse_args(["--restart", text])

    @pytest.mark.parametrize(
        "args, expected",
        [
            ([], RestartPolicy(RESTART_LUBY, 100)),  # no option: the example's default stays
            (["--restart", "none"], RestartPolicy()),
        ],
    )
    def test_solver_kwargs_from_args_restart(self, args: list[str], expected: RestartPolicy) -> None:
        parser = DefaultArgumentParser()
        kwargs = solver_kwargs_from_args(parser.parse_args(args), restart_policy=RestartPolicy(RESTART_LUBY, 100))
        assert kwargs["restart_policy"] == expected

    @pytest.mark.parametrize("args, expected", [([], None), (["--time-limit", "1500"], 1.5)])
    def test_timeout_from_args(self, args: list[str], expected: float | None) -> None:
        assert timeout_from_args(DefaultArgumentParser().parse_args(args)) == expected

    def test_run_solver_stops_at_the_time_limit(self) -> None:
        # queens(12) has 14,200 solutions: far more than a search of a few milliseconds finds, with or without the JIT
        args = DefaultArgumentParser().parse_args(
            ["--find-all", "--time-limit", "5", "--no-display-solutions", "--no-display-stats"]
        )
        solver = BacktrackSolver(QueensProblem(12))
        run_solver(solver, args)
        assert solver.interrupted
        assert solver.statistics[STATS_IDX_SOLUTION_NB] < 14_200

    def test_solver_kwargs_from_args_defaults(self) -> None:
        parser = DefaultArgumentParser()
        args = parser.parse_args([])
        kwargs = solver_kwargs_from_args(args, searches=[Search(dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)])
        assert kwargs == {"searches": [Search(dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)]}
