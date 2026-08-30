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

from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC


class TestDefaultArgumentParser:
    @pytest.mark.parametrize(
        "args, expected_args",
        [
            (["--consistency-algorithm", "BC"], {"consistency_algorithm": "BC"}),
            (["--display-solutions"], {"display_solutions": True}),
            (["--display-stats"], {"display_stats": True}),
            (["--dom-heuristic", "MIN_VALUE"], {"dom_heuristic": "MIN_VALUE"}),
            (["--find-all"], {"find_all": True}),
            (["--log-level", "INFO"], {"log_level": "INFO"}),
            (["--optimization-mode", "RESET"], {"optimization_mode": "RESET"}),
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
        kwargs = solver_kwargs_from_args(args, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)
        assert kwargs["consistency_algorithm"] == CONSISTENCY_ALG_BC
        assert kwargs["var_heuristic"] == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert kwargs["dom_heuristic"] == DOM_HEURISTIC_MIN_VALUE  # the CLI overrides the programmatic default

    def test_solver_kwargs_from_args_defaults(self) -> None:
        parser = DefaultArgumentParser()
        args = parser.parse_args([])
        kwargs = solver_kwargs_from_args(args, dom_heuristic=DOM_HEURISTIC_SPLIT_LOW)
        assert kwargs == {"dom_heuristic": DOM_HEURISTIC_SPLIT_LOW}
