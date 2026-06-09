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
"""
The ``fzn-nucs`` executable that MiniZinc invokes: it reads a FlatZinc file, solves it with NuCS and
streams the solutions on stdout.
"""

import argparse
import sys
from typing import List, Optional

from nucs.fzn.errors import FznError
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Builds the command-line argument parser for the FlatZinc solver interface.

    :return: the argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog="fzn-nucs", description="Solve a FlatZinc model with NuCS")
    parser.add_argument("fzn", help="the FlatZinc (.fzn) file to solve")
    parser.add_argument("-a", "--all-solutions", action="store_true", help="print all solutions")
    parser.add_argument("-n", "--num-solutions", type=int, default=None, help="stop after this many solutions")
    parser.add_argument("-s", "--statistics", action="store_true", help="print statistics to stderr")
    # Accepted and ignored in v1 for compatibility with the FlatZinc solver interface.
    parser.add_argument("-f", "--free-search", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-p", "--parallel", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-t", "--time-limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-r", "--random-seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the ``fzn-nucs`` executable.

    :param argv: the command-line arguments, defaults to sys.argv
    :type argv: Optional[List[str]]

    :return: the process exit code
    :rtype: int
    """
    args = build_arg_parser().parse_args(argv)
    try:
        with open(args.fzn, "r") as f:
            text = f.read()
        model = build_model(parse(text))
        run(
            model,
            sys.stdout,
            sys.stderr,
            all_solutions=args.all_solutions,
            num_solutions=args.num_solutions,
            statistics=args.statistics,
        )
    except FznError as e:
        sys.stderr.write(f"fzn-nucs: {e}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
