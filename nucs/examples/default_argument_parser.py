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
        self.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
        self.add_argument("--processors", type=int, default=1)
        self.add_argument("--progress_bar", type=bool, action=argparse.BooleanOptionalAction, default=False)
        self.add_argument("--shaving", type=bool, action=argparse.BooleanOptionalAction, default=False)
        self.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
