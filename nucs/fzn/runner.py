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

import logging
import signal
import socket
import threading
from typing import TextIO

from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import FznModel
from nucs.fzn.output import print_search_complete, print_solution, print_unknown, print_unsatisfiable
from nucs.fzn.search_annotations import free_search_heuristics, search_heuristics, search_options
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.restarts import Restarts
from nucs.solvers.solver import OPTIM_PRUNE

logger = logging.getLogger(__name__)


def run(
    model: FznModel,
    out: TextIO,
    all_solutions: bool = False,
    num_solutions: int | None = None,
    statistics: bool = False,
    output_mode: str = "item",
    output_objective: bool = False,
    intermediate_solutions: bool = False,
    time_limit_ms: int | None = None,
    stop_on_sigterm: bool = False,
    last_conflict: bool | None = None,
    free_search: bool = False,
    restart: Restarts | None = None,
) -> None:
    """
    Solves the model and writes the FlatZinc solution stream.

    Everything the FlatZinc interface defines -- solutions, the terminators and the statistics comments --
    goes to ``out``; stderr is left to the caller for genuine diagnostics.

    :param model: the built model
    :type model: FznModel
    :param out: the solution output stream
    :type out: TextIO
    :param all_solutions: whether to enumerate every solution (satisfy) or stream the improving ones
        (optimization)
    :type all_solutions: bool
    :param num_solutions: the maximum number of solutions to print (satisfy only), or None for one
    :type num_solutions: Optional[int]
    :param statistics: whether to print solver statistics
    :type statistics: bool
    :param output_mode: the solution output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param output_objective: whether to include the objective value in each solution (optimization only)
    :type output_objective: bool
    :param intermediate_solutions: whether to stream the improving solutions of an optimization problem
    :type intermediate_solutions: bool
    :param time_limit_ms: the wall-clock budget in milliseconds, or None for an unbounded search
    :type time_limit_ms: Optional[int]
    :param stop_on_sigterm: whether SIGTERM stops the search as a time limit would, which installs a process-wide
        signal handler and so is for the executable only
    :type stop_on_sigterm: bool
    :param last_conflict: whether the search branches first on the variable of the last refuted decision, or None for
        the mode's default (see search_options)
    :type last_conflict: Optional[bool]
    :param free_search: whether to ignore how the search annotations branch and use dom/wdeg instead
        (see free_search_heuristics)
    :type free_search: bool
    :param restart: the restart policy, or None for the mode's default (see search_options)
    :type restart: Optional[Restarts]
    """
    # Resolve the objective before constructing the solver, since the solver snapshots the domains on init.
    objective_var = None
    if model.solve.kind in ("minimize", "maximize"):
        if model.solve.objective is None:
            raise FznUnsupportedError("an optimization objective is required")
        objective_var = model.var_index_of(model.solve.objective)
    last_conflict, restarts = search_options(model, free_search, last_conflict, restart)
    solver = BacktrackSolver(
        model.problem,
        searches=free_search_heuristics(model) if free_search else search_heuristics(model),
        log_level="ERROR",
        restarts=restarts,
        last_conflict=last_conflict,
    )
    if stop_on_sigterm:
        _interrupt_on_sigterm(solver)
    timeout = None if time_limit_ms is None else time_limit_ms / 1000
    if model.solve.kind == "satisfy":
        _run_satisfy(model, solver, out, all_solutions, num_solutions, output_mode, timeout)
    else:
        assert objective_var is not None
        _run_optimize(
            model,
            solver,
            objective_var,
            out,
            output_mode,
            output_objective,
            all_solutions or intermediate_solutions or timeout is not None,
            timeout,
        )
    if statistics:
        _print_statistics(solver, out)


def _interrupt_on_sigterm(solver: BacktrackSolver) -> None:
    """
    Makes SIGTERM interrupt the search, so that the run still prints its best solution, its terminator and
    its statistics.

    That is how MiniZinc enforces its time limit: SIGTERM about a second after the limit, SIGKILL a second
    later. Left to the default action, SIGTERM kills the process with nothing printed. A Python handler alone
    would not do either: it runs only once the main thread is back in the interpreter, which a search in
    compiled code may not be before SIGKILL. The wakeup fd is written by the C-level handler as soon as the
    signal arrives, so a thread blocked on it interrupts the solver at once; the Python handler, whenever it
    runs, merely does the same.

    :param solver: the solver to interrupt
    :type solver: BacktrackSolver
    """
    reader, writer = socket.socketpair()
    writer.setblocking(False)
    signal.set_wakeup_fd(writer.fileno())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: solver.interrupt())

    def watch(reader: socket.socket, _writer: socket.socket) -> None:
        # the wakeup fd receives the number of every signal that has a Python handler, SIGINT included
        while reader.recv(1)[0] != signal.SIGTERM:
            pass
        solver.interrupt()

    # the thread holds the writer too: collected, it would close the wakeup fd along with it
    threading.Thread(target=watch, args=(reader, writer), name="fzn-nucs-sigterm", daemon=True).start()


def _run_optimize(
    model: FznModel,
    solver: BacktrackSolver,
    objective_var: int,
    out: TextIO,
    output_mode: str,
    output_objective: bool,
    intermediate_solutions: bool,
    timeout: float | None,
) -> None:
    """
    Prints the optimum of an optimization problem, or the whole sequence of improving solutions.

    The FlatZinc interface prints only the optimal solution by default; ``-a`` and ``-i`` are what ask for
    the intermediate ones. When they are given, each improving solution is printed as soon as it is found
    and the last of them is the optimum. The search-complete marker is printed once the optimum has been
    proven, and the unsatisfiable marker when no solution exists at all.

    A time limit also turns streaming on, whatever the flags say. Printing only at the end is safe exactly
    when the search is allowed to finish; under a deadline it is not, because the run may be ended from
    outside -- MiniZinc enforces its own limit with SIGTERM, then SIGKILL -- and a kill that lands before the
    end would lose a best solution that had never been printed. Streaming costs nothing there: MiniZinc keeps
    the last solution it received.

    The search runs in ``OPTIM_PRUNE`` mode: the tightened objective bound is applied to the choice points
    and the search resumes where it was, instead of restarting from the initial domains after every
    improving solution.

    :param model: the built model
    :type model: FznModel
    :param solver: the solver
    :type solver: BacktrackSolver
    :param objective_var: the NuCS variable index of the objective
    :type objective_var: int
    :param out: the solution output stream
    :type out: TextIO
    :param output_mode: the solution output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param output_objective: whether to include the objective value in each solution
    :type output_objective: bool
    :param intermediate_solutions: whether to print every improving solution rather than only the optimum
    :type intermediate_solutions: bool
    :param timeout: the wall-clock budget in seconds, or None for an unbounded search
    :type timeout: Optional[float]
    """
    solutions = solver.optimize(
        objective_var,
        bound=DOMAIN_MIN if model.solve.kind == "minimize" else DOMAIN_MAX,
        mode=OPTIM_PRUNE,
        timeout=timeout,
    )
    best = None
    printed = False
    for solution in solutions:
        if intermediate_solutions:
            _print_optimization_solution(model, solution, objective_var, out, output_mode, output_objective)
            printed = True
        else:
            best = solution  # a copy of the domains (get_solution), which the search does not overwrite
    # the solver stops the iteration itself when the budget runs out, so a proof is what it did not report
    proven = not solver.timed_out
    if best is not None:
        _print_optimization_solution(model, best, objective_var, out, output_mode, output_objective)
        printed = True
    if printed:
        if proven:
            print_search_complete(out)
    elif proven:
        print_unsatisfiable(out)
    else:
        print_unknown(out)
    out.flush()


def _print_optimization_solution(
    model: FznModel,
    solution: NDArray,
    objective_var: int,
    out: TextIO,
    output_mode: str,
    output_objective: bool,
) -> None:
    """
    Prints one solution of an optimization problem, with its objective value when requested.

    :param model: the built model
    :type model: FznModel
    :param solution: the solution
    :type solution: NDArray
    :param objective_var: the NuCS variable index of the objective
    :type objective_var: int
    :param out: the solution output stream
    :type out: TextIO
    :param output_mode: the solution output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param output_objective: whether to include the objective value
    :type output_objective: bool
    """
    objective_value = int(solution[objective_var]) if output_objective else None
    print_solution(model, solution, out, output_mode, objective_value)
    out.flush()


def _run_satisfy(
    model: FznModel,
    solver: BacktrackSolver,
    out: TextIO,
    all_solutions: bool,
    num_solutions: int | None,
    output_mode: str,
    timeout: float | None,
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
    :param output_mode: the solution output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param timeout: the wall-clock budget in seconds, or None for an unbounded search
    :type timeout: Optional[float]
    """
    limit = None if all_solutions else (num_solutions if num_solutions is not None else 1)
    found = False
    exhausted = False
    for count, solution in enumerate(solver.solve(timeout=timeout), start=1):
        print_solution(model, solution, out, output_mode)
        found = True
        if limit is not None and count >= limit:
            break
    else:
        # the iteration ran out on its own: the space is exhausted unless the budget stopped it first
        exhausted = not solver.timed_out
    if not found:
        # Nothing found and the space was not exhausted is precisely what the unknown marker reports.
        print_unknown(out) if solver.timed_out else print_unsatisfiable(out)
    elif exhausted:
        print_search_complete(out)
    out.flush()


def _print_statistics(solver: BacktrackSolver, out: TextIO) -> None:
    """
    Prints solver statistics as MiniZinc-style comment lines on the solution stream.

    The FlatZinc interface puts statistics on standard output as comments, not on stderr: written anywhere
    else, ``minizinc -s --solver nucs`` never sees them. This single block comes after the search, which the
    specification allows as concluding output.

    :param solver: the solver
    :type solver: BacktrackSolver
    :param out: the solution output stream
    :type out: TextIO
    """
    out.writelines(f"%%%mzn-stat: {key}={value}\n" for key, value in solver.get_statistics_as_dictionary().items())
    out.write("%%%mzn-stat-end\n")
    out.flush()
