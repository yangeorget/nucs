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
from typing import Any

from numpy.typing import NDArray

from nucs.constants import LOG_LEVELS, OPTIM_MODES, OPTIM_RESET
from nucs.heuristics.heuristics import DOM_HEURISTICS, VAR_HEURISTICS
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALGS
from nucs.solvers.solver import Solver


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        # the choices below are computed when the parser is built:
        # custom algorithms/heuristics registered beforehand are also selectable from the CLI
        self.add_argument(
            "--consistency-algorithm",
            help="set the consistency algorithm",
            choices=sorted(CONSISTENCY_ALGS),
        )
        self.add_argument(
            "--cp-max-height",
            help="set the maximal height of the choice points stack",
            type=int,
        )
        self.add_argument(
            "--dom-heuristic",
            help="set the domain heuristic",
            choices=sorted(DOM_HEURISTICS),
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
            choices=sorted(VAR_HEURISTICS),
        )


def solver_kwargs_from_args(args: Namespace, **defaults: Any) -> dict[str, Any]:
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
        "consistency_algorithm": None
        if args.consistency_algorithm is None
        else CONSISTENCY_ALGS[args.consistency_algorithm],
        "stks_max_height": args.cp_max_height,
        "var_heuristic": None if args.var_heuristic is None else VAR_HEURISTICS[args.var_heuristic],
        "dom_heuristic": None if args.dom_heuristic is None else DOM_HEURISTICS[args.dom_heuristic],
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
        solution = next(solver.solve(), None)  # type: ignore
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            solver.problem.print_solution(solution)


def run_optimizer(
    solver: Solver,
    args: Namespace,
    objective: int,
    maximize: bool = False,
    default_mode: str = OPTIM_RESET,
) -> NDArray | None:
    """
    Optimizes a variable with the solver according to the CLI arguments and returns the optimal solution.

    :param solver: the solver
    :type solver: Solver
    :param args: the CLI arguments
    :type args: Namespace
    :param objective: the index of the variable to optimize
    :type objective: int
    :param maximize: whether to maximize the objective, minimize it otherwise, defaults to False
    :type maximize: bool
    :param default_mode: the optimization mode used when none is set on the command line, defaults to OPTIM_RESET
    :type default_mode: str

    :return: the optimal solution if it exists or None
    :rtype: Optional[NDArray]
    """
    mode = args.optimization_mode or default_mode
    solution = (solver.maximize if maximize else solver.minimize)(objective, mode=mode)
    if args.display_stats:
        solver.print_statistics()
    if args.display_solutions:
        solver.problem.print_solution(solution)
    return solution
