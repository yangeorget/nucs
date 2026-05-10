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
from argparse import Namespace
from typing import Any, Dict

from nucs.constants import LOG_LEVELS, OPTIM_MODES
from nucs.solvers.solver import Solver


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


def solver_kwargs_from_args(args: Namespace, **defaults: Any) -> Dict[str, Any]:
    """
    Builds a dict of BacktrackSolver kwargs, with CLI args overriding the given defaults.

    :param args: the CLI arguments
    :type args: Namespace
    :param defaults: kwargs to be passed to BacktrackSolver, overridden by any non-None CLI value
    :type defaults: Any

    :return: a dict of kwargs
    :rtype: Dict[str, Any]
    """
    overrides = {
        "consistency_algorithm": args.consistency_algorithm,
        "stks_max_height": args.cp_max_height,
        "var_heuristic": args.var_heuristic,
        "dom_heuristic": args.dom_heuristic,
        "log_level": args.log_level,
    }
    return {**defaults, **{k: v for k, v in overrides.items() if v is not None}}


def run_solver(solver: Solver, args: Namespace) -> None:
    """
    Runs the solver according to the CLI arguments.

    :param solver: the solver
    :type solver: Solver
    :param args: the CLI arguments
    :type args: Namespace
    """
    if args.find_all:
        solutions = solver.find_all()
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            for solution in solutions:
                solver.problem.print_solution(solution)
    else:
        solution = next(solver.solve())
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            solver.problem.print_solution(solution)
