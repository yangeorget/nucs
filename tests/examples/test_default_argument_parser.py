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
from typing import Dict, Any

import pytest

from nucs.examples.default_argument_parser import DefaultArgumentParser


class TestDefaultArgumentParser:
    @pytest.mark.parametrize(
        "args, expected_args",
        [
            (["--consistency", "0"], {"consistency": 0}),
            (["--consistency", "1"], {"consistency": 1}),
            (["--cp-max-height", "512"], {"cp_max_height": 512}),
            (["--display-solutions"], {"display_solutions": True}),
            (["--display-stats"], {"display_stats": True}),
            (["--find-all"], {"find_all": True}),
            (["--ff"], {"ff": True}),
            (["--log-level", "INFO"], {"log_level": "INFO"}),
            (["--optimization-mode", "RESET"], {"optimization_mode": "RESET"}),
            (["--processors", "4"], {"processors": 4}),
            (["--symmetry-breaking"], {"symmetry_breaking": True}),
        ],
    )
    def test_parse_args(self, args: str, expected_args: Dict[str, Any]) -> None:
        parser = DefaultArgumentParser()
        parsed_args = parser.parse_args(args)
        for arg, value in expected_args.items():
            assert getattr(parsed_args, arg) == value
