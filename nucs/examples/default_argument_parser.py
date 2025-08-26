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

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS, OPTIM_MODES, OPTIM_RESET
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument(
            "--consistency",
            help="set the consistency algorithm (0 is for BC, 1 for BC+shaving)",
            type=int,
            default=CONSISTENCY_ALG_BC,
        )
        self.add_argument(
            "--cp-max-height",
            help="set the maximal height of the choice points stack (default is 512)",
            type=int,
            default=512,
        )
        self.add_argument(
            "--display-solutions",
            help="display the solution(s)",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--display-stats",
            help="display the statistics",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--find-all",
            help="find all solutions",
            type=bool,
            action=argparse.BooleanOptionalAction,
            efault=True,
        )
        self.add_argument(
            "--log-level",
            help="set the log level",
            choices=LOG_LEVELS,
            default=LOG_LEVEL_INFO,
        )
        self.add_argument(
            "--optimization-mode",
            help="set the optimization mode",
            choices=OPTIM_MODES,
            default=OPTIM_RESET,
        )
        self.add_argument(
            "--processors",
            help="set the number of processors",
            type=int,
            default=1,
        )
        self.add_argument(
            "--symmetry-breaking",
            help="add symmetry breaking constraints",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
