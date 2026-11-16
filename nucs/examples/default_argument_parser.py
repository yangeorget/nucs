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
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, LOG_LEVELS
from nucs.heuristics.heuristics import DOM_HEURISTICS, VAR_HEURISTICS
from nucs.solvers.backtrack.consistency_algorithms import CONSISTENCY_ALGS
from nucs.solvers.restart_policies import RestartPolicy, parse_restart
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.solver import OPTIM_MODES, OPTIM_RESET, Solver


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
        # no default: an example that turns last-conflict on keeps it unless the command line says otherwise
        self.add_argument(
            "--last-conflict",
            help="branch first on the variable of the last refuted decision, as long as it is unbound",
            action=argparse.BooleanOptionalAction,
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
        # no default: an example that sets a restart policy keeps it unless the command line says otherwise
        self.add_argument(
            "--restart",
            help="restart after a number of failures: POLICY[,SCALE[,BASE]], the policy being luby, geometric, linear, "
            "constant or none, e.g. luby,100",
            metavar="POLICY[,SCALE[,BASE]]",
            type=restart_arg,
        )
        self.add_argument(
            "--symmetry-breaking",
            help="add symmetry breaking constraints",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.add_argument(
            "--time-limit",
            help="stop the search after this many milliseconds; an optimization reports the best solution found",
            type=int,
            metavar="MS",
        )
        self.add_argument(
            "--var-heuristic",
            help="set the variable heuristic",
            choices=sorted(VAR_HEURISTICS),
        )


def restart_arg(text: str) -> RestartPolicy:
    """
    Converts the --restart argument, so that a wrong one is reported as a usage error with its reason.

    :param text: the argument, POLICY[,SCALE[,BASE]]
    :type text: str

    :return: the restart policy
    :rtype: RestartPolicy
    """
    try:
        return parse_restart(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def solver_kwargs_from_args(
    args: Namespace, searches: Sequence[Search] = SEARCHES_DEFAULT, **defaults: Any
) -> dict[str, Any]:
    """
    Builds a dict of BacktrackSolver kwargs, with CLI args overriding the given defaults.

    A variable or domain heuristic given on the command line replaces that of every search, which keeps its
    parameters.

    :param args: the CLI arguments
    :type args: Namespace
    :param searches: the searches of the example, defaults to SEARCHES_DEFAULT
    :type searches: Sequence[Search]
    :param defaults: other kwargs to be passed to BacktrackSolver, overridden by any non-None CLI value
    :type defaults: Any

    :return: a dict of kwargs
    :rtype: Dict[str, Any]
    """
    if args.var_heuristic is not None:
        searches = [replace(search, var_heuristic=VAR_HEURISTICS[args.var_heuristic]) for search in searches]
    if args.dom_heuristic is not None:
        searches = [replace(search, dom_heuristic=DOM_HEURISTICS[args.dom_heuristic]) for search in searches]
    overrides = {
        "consistency_algorithm": None
        if args.consistency_algorithm is None
        else CONSISTENCY_ALGS[args.consistency_algorithm],
        "last_conflict": args.last_conflict,  # None, True or False: --no-last-conflict must override a True default
        "log_level": args.log_level,
        "restart_policy": args.restart,
    }
    return {**defaults, "searches": searches, **{k: v for k, v in overrides.items() if v is not None}}


def timeout_from_args(args: Namespace) -> float | None:
    """
    Returns the search budget that --time-limit gives, in seconds as the solver takes it.

    :param args: the CLI arguments
    :type args: Namespace

    :return: the budget in seconds, or None for an unbounded search
    :rtype: Optional[float]
    """
    return None if args.time_limit is None else args.time_limit / 1000


def run_solver(solver: Solver, args: Namespace) -> None:
    """
    Runs the solver according to the CLI arguments.

    :param solver: the solver
    :type solver: Solver
    :param args: the CLI arguments
    :type args: Namespace
    """
    if args.find_all:
        solutions = solver.find_all(timeout=timeout_from_args(args))
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            for solution in solutions:
                solver.problem.print_solution(solution)
    else:
        solution = next(solver.solve(timeout=timeout_from_args(args)), None)  # type: ignore
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
    solution = solver.find_best(
        objective, bound=DOMAIN_MAX if maximize else DOMAIN_MIN, mode=mode, timeout=timeout_from_args(args)
    )
    if args.display_stats:
        solver.print_statistics()
    if args.display_solutions:
        solver.problem.print_solution(solution)
    return solution
