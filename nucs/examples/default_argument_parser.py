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
import argparse

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument(
            "--all", help="find all solutions", type=bool, action=argparse.BooleanOptionalAction, default=True
        )
        self.add_argument(
            "--log_level",
            help="set the log level",
            choices=LOG_LEVELS,
            default=LOG_LEVEL_INFO,
        )
        self.add_argument(
            "--processors",
            help="set the number of processors",
            type=int,
            default=1,
        )
        self.add_argument(
            "--progress_bar",
            help="show a progress bar",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        self.add_argument(
            "--display",
            help="display the solution(s)",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--stats",
            help="display the statistics",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--shaving",
            help="use shaving together with bound consistency",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        self.add_argument(
            "--symmetry_breaking",
            help="add symmetry breaking constraints",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
