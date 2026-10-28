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

from nucs.fzn.__main__ import build_arg_parser
from nucs.solvers.restarts import RESTART_LUBY, RestartPolicy


class TestMain:
    def test_build_arg_parser_cli_parses_output_options(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        args = build_arg_parser().parse_args(["model.fzn", "--output-mode", "json", "--output-objective"])
        assert args.output_mode == "json"
        assert args.output_objective is True

    def test_parse_args_cli_parses_time_limit(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        assert build_arg_parser().parse_args(["model.fzn", "-t", "1500"]).time_limit == 1500
        assert build_arg_parser().parse_args(["model.fzn"]).time_limit is None

    def test_parse_args_cli_parses_intermediate_solutions(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        assert build_arg_parser().parse_args(["model.fzn", "-i"]).intermediate_solutions is True
        assert build_arg_parser().parse_args(["model.fzn"]).intermediate_solutions is False

    def test_parse_args_cli_output_mode_defaults_to_item(self) -> None:
        from nucs.fzn.__main__ import build_arg_parser

        args = build_arg_parser().parse_args(["model.fzn"])
        assert args.output_mode == "item"
        assert args.output_objective is False

    def test_main_free_search_and_restart_flags(self) -> None:
        args = build_arg_parser().parse_args(["-f", "--restart", "luby,500", "model.fzn"])
        assert args.free_search
        assert args.restart == RestartPolicy(RESTART_LUBY, 500)
        with pytest.raises(SystemExit):  # a wrong policy is a usage error
            build_arg_parser().parse_args(["--restart", "fibonacci", "model.fzn"])

    def test_main_last_conflict_flag(self) -> None:
        assert build_arg_parser().parse_args(["--last-conflict", "model.fzn"]).last_conflict is True
        assert build_arg_parser().parse_args(["--no-last-conflict", "model.fzn"]).last_conflict is False
        assert build_arg_parser().parse_args(["model.fzn"]).last_conflict is None  # the mode's default
