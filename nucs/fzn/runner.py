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
from collections.abc import Sequence
from typing import TextIO

from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import FznModel
from nucs.fzn.output import print_search_complete, print_solution, print_unknown, print_unsatisfiable
from nucs.fzn.parser import Ann, Id
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_RANDOM_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_DOM_WDEG,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_GREATEST_DOMAIN,
    VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
    VAR_HEURISTIC_MAX_REGRET,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.restarts import (
    NO_RESTARTS,
    RESTART_CONSTANT,
    RESTART_GEOMETRIC,
    RESTART_LINEAR,
    RESTART_LUBY,
    Restarts,
)
from nucs.solvers.search import DEFAULT_SEARCHES, Search
from nucs.solvers.solver import OPTIM_PRUNE

logger = logging.getLogger(__name__)

# FlatZinc variable-selection annotations mapped to NuCS variable heuristics; unlisted ones
# (occurrence, most_constrained, ...) fall back to the default, with a warning -- silently
# solving a different search than the model asked for reads as slowness rather than as a missing feature.
_VAR_HEURISTICS = {
    "input_order": VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    "first_fail": VAR_HEURISTIC_SMALLEST_DOMAIN,
    "dom_w_deg": VAR_HEURISTIC_DOM_WDEG,
    "anti_first_fail": VAR_HEURISTIC_GREATEST_DOMAIN,
    "max_regret": VAR_HEURISTIC_MAX_REGRET,
    "smallest": VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
    "largest": VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
}
# FlatZinc value-selection annotations mapped to NuCS domain heuristics; likewise for the unlisted ones
# (indomain_interval, outdomain_min, ...).
_DOM_HEURISTICS = {
    "indomain_min": DOM_HEURISTIC_MIN_VALUE,
    "indomain_max": DOM_HEURISTIC_MAX_VALUE,
    "indomain_median": DOM_HEURISTIC_MID_VALUE,
    "indomain_random": DOM_HEURISTIC_RANDOM_VALUE,
    "indomain_split": DOM_HEURISTIC_SPLIT_LOW,
    "indomain_reverse_split": DOM_HEURISTIC_SPLIT_HIGH,
}


# the MiniZinc restart annotations whose only argument is the scale
_SCALED_RESTARTS = {
    "restart_constant": RESTART_CONSTANT,
    "restart_linear": RESTART_LINEAR,
    "restart_luby": RESTART_LUBY,
}


def restart_policy_of(model: FznModel) -> Restarts:
    """
    Translates the restart annotation of the solve item into a NuCS restart policy. A restart limit counts failures,
    as Gecode's do.

    :param model: the built model
    :type model: FznModel

    :return: the restart policy, for BacktrackSolver; no restart when the model asks for none, or asks for one with
             arguments NuCS cannot use, which is logged as a warning
    :rtype: Restarts
    """
    for annotation in model.solve.annotations:
        args = annotation.args
        try:
            if annotation.name == "restart_none":
                return NO_RESTARTS
            if annotation.name in _SCALED_RESTARTS and len(args) == 1 and isinstance(args[0], int):
                return Restarts(_SCALED_RESTARTS[annotation.name], args[0])
            if (
                annotation.name == "restart_geometric"
                and len(args) == 2
                and isinstance(args[0], (int, float))
                and isinstance(args[1], int)
            ):
                return Restarts(RESTART_GEOMETRIC, args[1], float(args[0]))
        except ValueError as e:
            logger.warning(f"{e}, searching without restarts")
            return NO_RESTARTS
        if annotation.name.startswith("restart_"):
            logger.warning(f"Unsupported restart annotation {annotation.name}{args}, searching without restarts")
            return NO_RESTARTS
    return NO_RESTARTS


# The free search (-f) defaults, measured on 41 challenge instances in 60 s against the models' own searches:
# last-conflict won 9 and lost 0 against the same free search without it; Luby restarts at scale 500 won 10 and
# lost 3 (proofs that got slower) against the free search without restarts.
FREE_SEARCH_LAST_CONFLICT = True
FREE_SEARCH_RESTART = Restarts(RESTART_LUBY, 500)


def search_options(
    model: FznModel, free_search: bool, last_conflict: bool | None, restart: Restarts | None
) -> tuple[bool, Restarts]:
    """
    Resolves last-conflict and the restart policy from the mode and the command line. What the command line gives
    wins. Otherwise the free search takes its defaults and ignores the model's restart annotation, as it ignores the
    rest of its search annotation; the fixed search takes no last-conflict and follows the restart annotation.

    :param model: the built model
    :type model: FznModel
    :param free_search: whether the search is free (-f)
    :type free_search: bool
    :param last_conflict: last-conflict as the command line gives it, or None for the mode's default
    :type last_conflict: Optional[bool]
    :param restart: the restart policy as the command line gives it, or None for the mode's default
    :type restart: Optional[Restarts]

    :return: whether to use last-conflict, and the restart policy
    :rtype: Tuple[bool, Restarts]
    """
    if last_conflict is None:
        last_conflict = FREE_SEARCH_LAST_CONFLICT if free_search else False
    if restart is None:
        restart = FREE_SEARCH_RESTART if free_search else restart_policy_of(model)
    return last_conflict, restart


def parse_restart(text: str) -> Restarts:
    """
    Parses a restart policy given on the command line as ``POLICY[,SCALE[,BASE]]``, such as ``luby,500`` or
    ``geometric,100,1.5``. The scale defaults to 100 and the base to 1.5, as in BacktrackSolver.

    :param text: the policy, one of RESTART_POLICIES, then optionally its scale and its base
    :type text: str

    :return: the restart policy
    :rtype: Restarts
    """
    parts = text.split(",")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Expected POLICY[,SCALE[,BASE]], not {text!r}")
    policy = parts[0].strip().lower()
    scale = int(parts[1]) if len(parts) > 1 else 100
    base = float(parts[2]) if len(parts) > 2 else 1.5
    return Restarts(policy, scale, base)  # raises on a wrong policy, scale or base


def search_heuristics(model: FznModel) -> Sequence[Search]:
    """
    Translates the first ``int_search``/``bool_search``/``seq_search`` annotation on the solve item into a
    NuCS sequential search: one :class:`Search` per nested search, each keeping its own variable and value
    selectors. A ``seq_search`` becomes the ordered list of its nested searches (flattened recursively, since
    MiniZinc nests a ``seq_search`` when decomposing e.g. ``set_search``); a single ``int_search`` /
    ``bool_search`` becomes a one-element list. Each variable is assigned to the first search that lists it,
    and a trailing catch-all search over the remaining variables (NuCS defaults) makes the search ground the
    whole problem. Unknown variable/value selectors fall back to the NuCS defaults, and each one is
    logged as a warning.

    :param model: the built model
    :type model: FznModel

    :return: the ordered searches, or DEFAULT_SEARCHES when there is no supported search annotation
    :rtype: Sequence[Search]
    """
    for annotation in model.solve.annotations:
        nested = _flatten_searches(model, annotation)
        if nested:
            searches: list[Search] = []
            seen: set = set()
            for variables, var_heuristic, dom_heuristic in nested:
                group = [v for v in variables if v not in seen]  # a variable belongs to the first search listing it
                seen.update(group)
                if group:
                    searches.append(Search(group, var_heuristic, [[]], dom_heuristic, [[]]))
            remaining = [v for v in range(model.problem.domain_nb) if v not in seen]
            if remaining:  # ground every remaining variable with the defaults
                searches.append(Search(remaining))
            return searches or DEFAULT_SEARCHES
    return DEFAULT_SEARCHES


def free_search_heuristics(model: FznModel) -> list[Search]:
    """
    Builds the free search (``-f``): the model's search annotations only say which variables it decides, and both
    the order and the values are left to NuCS. The variables the annotations name come first, in one dom/wdeg search,
    since they are the modeller's decisions; the remaining variables follow, in a second one, so that the search
    grounds the whole problem. A model with no search annotation gets one dom/wdeg search over all its variables.

    :param model: the built model
    :type model: FznModel

    :return: the ordered list of searches
    :rtype: List[Search]
    """
    annotated: list[int] = []
    seen: set = set()
    for annotation in model.solve.annotations:
        for variable in _annotated_variables(model, annotation):
            if variable not in seen:
                seen.add(variable)
                annotated.append(variable)
    remaining = [v for v in range(model.problem.domain_nb) if v not in seen]
    return [Search(variables, VAR_HEURISTIC_DOM_WDEG) for variables in (annotated, remaining) if variables]


def _annotated_variables(model: FznModel, annotation: Ann) -> list[int]:
    """
    Returns the variables a search annotation branches on, in order, recursing into ``seq_search``. Unlike
    _single_search it reads no selector, so the selectors a free search ignores raise no warning.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: the variables, possibly with repetitions
    :rtype: List[int]
    """
    if annotation.name == "seq_search" and annotation.args and isinstance(annotation.args[0], list):
        return [v for item in annotation.args[0] if isinstance(item, Ann) for v in _annotated_variables(model, item)]
    if annotation.name in ("int_search", "bool_search") and annotation.args:
        return model.var_list_of(annotation.args[0])
    return []


def _flatten_searches(model: FznModel, annotation: Ann) -> list[tuple[list[int], int, int]]:
    """
    Flattens a search annotation into an ordered list of (search variables, variable heuristic, domain
    heuristic) triples, recursing into nested ``seq_search`` annotations and dropping unsupported ones.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: the flattened list of search triples (empty when nothing is supported)
    :rtype: List[Tuple[List[int], int, int]]
    """
    if annotation.name == "seq_search" and annotation.args and isinstance(annotation.args[0], list):
        flattened: list[tuple[list[int], int, int]] = []
        for item in annotation.args[0]:
            if isinstance(item, Ann):
                flattened.extend(_flatten_searches(model, item))
        return flattened
    single = _single_search(model, annotation)
    return [single] if single is not None else []


def _single_search(model: FznModel, annotation: Ann) -> tuple[list[int], int, int] | None:
    """
    Translates a single ``int_search``/``bool_search`` annotation into a NuCS search configuration.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: a triple (search variables, variable heuristic, domain heuristic), or None when the annotation
             is not a supported search
    :rtype: Optional[Tuple[List[int], int, int]]
    """
    if annotation.name in ("int_search", "bool_search") and annotation.args:
        search_variables = model.var_list_of(annotation.args[0])
        var_heuristic = _var_heuristic_of(annotation.args[1] if len(annotation.args) > 1 else None)
        dom_heuristic = _dom_heuristic_of(annotation.args[2] if len(annotation.args) > 2 else None)
        return search_variables, var_heuristic, dom_heuristic
    return None


def _var_heuristic_of(term: object) -> int:
    """
    Returns the NuCS variable heuristic for a FlatZinc selector term, defaulting to first-not-instantiated.

    :param term: the selector term (an Id) or None
    :type term: object

    :return: the variable heuristic id
    :rtype: int
    """
    if isinstance(term, Id):
        if term.name not in _VAR_HEURISTICS:
            logger.warning(f"Unsupported variable selector {term.name}, branching as input_order instead")
        return _VAR_HEURISTICS.get(term.name, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED)
    return VAR_HEURISTIC_FIRST_NOT_INSTANTIATED


def _dom_heuristic_of(term: object) -> int:
    """
    Returns the NuCS domain heuristic for a FlatZinc selector term, defaulting to min-value.

    :param term: the selector term (an Id) or None
    :type term: object

    :return: the domain heuristic id
    :rtype: int
    """
    if isinstance(term, Id):
        if term.name not in _DOM_HEURISTICS:
            logger.warning(f"Unsupported value selector {term.name}, branching as indomain_min instead")
        return _DOM_HEURISTICS.get(term.name, DOM_HEURISTIC_MIN_VALUE)
    return DOM_HEURISTIC_MIN_VALUE


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
