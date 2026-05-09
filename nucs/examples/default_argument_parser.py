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
import argparse

from nucs.constants import LOG_LEVELS, OPTIM_MODES


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument(
            "--consistency-algorithm",
            help="set the consistency algorithm (0 is for BC, 1 for BC+shaving)",
            type=int,
        )
        self.add_argument(
            "--cp-max-height",
            help="set the maximal height of the choice points stack",
            type=int,
        )
        self.add_argument(
            "--dom-heuristic",
            help="set the domain heuristic",
            type=int,
        )
        self.add_argument(
            "--display-solutions",
            help="display the solution(s)",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--display-stats",
            help="display the statistics",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--find-all",
            help="find all solutions",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        self.add_argument(
            "--log-level",
            help="set the log level",
            choices=LOG_LEVELS,
        )
        self.add_argument(
            "--optimization-mode",
            help="set the optimization mode",
            choices=OPTIM_MODES,
        )
        self.add_argument(
            "--processors",
            help="set the number of processors",
            type=int,
        )
        self.add_argument(
            "--symmetry-breaking",
            help="add symmetry breaking constraints",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--var-heuristic",
            help="set the variable heuristic",
            type=int,
        )
