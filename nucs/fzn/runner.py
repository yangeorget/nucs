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
Drives a built :class:`FznModel` through a :class:`BacktrackSolver` and streams the solutions.
"""

from typing import Optional, TextIO

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import FznModel
from nucs.fzn.output import print_search_complete, print_solution, print_unsatisfiable
from nucs.solvers.backtrack_solver import BacktrackSolver


def run(
    model: FznModel,
    out: TextIO,
    err: TextIO,
    all_solutions: bool = False,
    num_solutions: Optional[int] = None,
    statistics: bool = False,
) -> None:
    """
    Solves the model and writes the FlatZinc solution stream.

    :param model: the built model
    :type model: FznModel
    :param out: the solution output stream
    :type out: TextIO
    :param err: the diagnostics/statistics stream
    :type err: TextIO
    :param all_solutions: whether to enumerate every solution (satisfy only)
    :type all_solutions: bool
    :param num_solutions: the maximum number of solutions to print (satisfy only), or None for one
    :type num_solutions: Optional[int]
    :param statistics: whether to print solver statistics to err
    :type statistics: bool
    """
    # Resolve the objective before constructing the solver, since the solver snapshots the domains on init.
    objective_var = None
    if model.solve.kind in ("minimize", "maximize"):
        if model.solve.objective is None:
            raise FznUnsupportedError("an optimization objective is required")
        objective_var = model.var_index_of(model.solve.objective)
    solver = BacktrackSolver(model.problem, log_level="ERROR")
    if model.solve.kind == "satisfy":
        _run_satisfy(model, solver, out, all_solutions, num_solutions)
    else:
        assert objective_var is not None
        if model.solve.kind == "minimize":
            solution = solver.minimize(objective_var)
        else:
            solution = solver.maximize(objective_var)
        if solution is None:
            print_unsatisfiable(out)
        else:
            print_solution(model, solution, out)
            print_search_complete(out)
    if statistics:
        _print_statistics(solver, err)


def _run_satisfy(
    model: FznModel,
    solver: BacktrackSolver,
    out: TextIO,
    all_solutions: bool,
    num_solutions: Optional[int],
) -> None:
    """
    Iterates satisfy solutions honoring the all/limit flags and prints the appropriate terminators.

    :param model: the built model
    :type model: FznModel
    :param solver: the solver
    :type solver: BacktrackSolver
    :param out: the solution output stream
    :type out: TextIO
    :param all_solutions: whether to enumerate every solution
    :type all_solutions: bool
    :param num_solutions: the maximum number of solutions, or None for one
    :type num_solutions: Optional[int]
    """
    limit = None if all_solutions else (num_solutions if num_solutions is not None else 1)
    count = 0
    found = False
    exhausted = True
    for solution in solver.solve():
        print_solution(model, solution, out)
        found = True
        count += 1
        if limit is not None and count >= limit:
            exhausted = False
            break
    if not found:
        print_unsatisfiable(out)
    elif exhausted:
        print_search_complete(out)


def _print_statistics(solver: BacktrackSolver, err: TextIO) -> None:
    """
    Prints solver statistics as MiniZinc-style comment lines.

    :param solver: the solver
    :type solver: BacktrackSolver
    :param err: the diagnostics stream
    :type err: TextIO
    """
    for key, value in solver.get_statistics_as_dictionary().items():
        err.write(f"%%%mzn-stat: {key}={value}\n")
    err.write("%%%mzn-stat-end\n")
