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

from typing import List, Optional, TextIO, Tuple

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.model import FznModel
from nucs.fzn.output import print_search_complete, print_solution, print_unsatisfiable
from nucs.fzn.parser import Id
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_GREATEST_DOMAIN,
    VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
    VAR_HEURISTIC_MAX_REGRET,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
)
from nucs.solvers.backtrack_solver import BacktrackSolver

# FlatZinc variable-selection annotations mapped to NuCS variable heuristics; unlisted ones
# (dom_w_deg, occurrence, most_constrained, ...) fall back to the default.
_VAR_HEURISTICS = {
    "input_order": VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    "first_fail": VAR_HEURISTIC_SMALLEST_DOMAIN,
    "anti_first_fail": VAR_HEURISTIC_GREATEST_DOMAIN,
    "max_regret": VAR_HEURISTIC_MAX_REGRET,
    "smallest": VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
    "largest": VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
}
# FlatZinc value-selection annotations mapped to NuCS domain heuristics.
_DOM_HEURISTICS = {
    "indomain_min": DOM_HEURISTIC_MIN_VALUE,
    "indomain_max": DOM_HEURISTIC_MAX_VALUE,
    "indomain_median": DOM_HEURISTIC_MID_VALUE,
    "indomain_split": DOM_HEURISTIC_SPLIT_LOW,
}


def search_heuristics(model: FznModel) -> Optional[Tuple[List[int], int, int]]:
    """
    Translates the first ``int_search``/``bool_search`` annotation on the solve item into a NuCS search
    configuration.

    The listed variables come first in the decision order (honoring ``input_order``), followed by every
    remaining variable so that the search always grounds the whole problem. Unknown variable/value
    selectors fall back to the NuCS defaults.

    :param model: the built model
    :type model: FznModel

    :return: a triple (decision variables, variable heuristic, domain heuristic), or None when there is no
             supported search annotation
    :rtype: Optional[Tuple[List[int], int, int]]
    """
    for annotation in model.solve.annotations:
        if annotation.name in ("int_search", "bool_search") and annotation.args:
            search_variables = model.var_list_of(annotation.args[0])
            var_heuristic = _var_heuristic_of(annotation.args[1] if len(annotation.args) > 1 else None)
            dom_heuristic = _dom_heuristic_of(annotation.args[2] if len(annotation.args) > 2 else None)
            seen = set(search_variables)
            decision_variables = search_variables + [v for v in range(model.problem.domain_nb) if v not in seen]
            return decision_variables, var_heuristic, dom_heuristic
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
        return _DOM_HEURISTICS.get(term.name, DOM_HEURISTIC_MIN_VALUE)
    return DOM_HEURISTIC_MIN_VALUE


def run(
    model: FznModel,
    out: TextIO,
    err: TextIO,
    all_solutions: bool = False,
    num_solutions: Optional[int] = None,
    statistics: bool = False,
    output_mode: str = "item",
    output_objective: bool = False,
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
    :param output_mode: the solution output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param output_objective: whether to include the objective value in each solution (optimization only)
    :type output_objective: bool
    """
    # Resolve the objective before constructing the solver, since the solver snapshots the domains on init.
    objective_var = None
    if model.solve.kind in ("minimize", "maximize"):
        if model.solve.objective is None:
            raise FznUnsupportedError("an optimization objective is required")
        objective_var = model.var_index_of(model.solve.objective)
    search = search_heuristics(model)
    if search is None:
        solver = BacktrackSolver(model.problem, log_level="ERROR")
    else:
        decision_variables, var_heuristic, dom_heuristic = search
        solver = BacktrackSolver(
            model.problem,
            decision_variables=decision_variables,
            var_heuristic=var_heuristic,
            dom_heuristic=dom_heuristic,
            log_level="ERROR",
        )
    if model.solve.kind == "satisfy":
        _run_satisfy(model, solver, out, all_solutions, num_solutions, output_mode)
    else:
        assert objective_var is not None
        if model.solve.kind == "minimize":
            solution = solver.minimize(objective_var)
        else:
            solution = solver.maximize(objective_var)
        if solution is None:
            print_unsatisfiable(out)
        else:
            objective_value = int(solution[objective_var]) if output_objective else None
            print_solution(model, solution, out, output_mode, objective_value)
            print_search_complete(out)
    if statistics:
        _print_statistics(solver, err)


def _run_satisfy(
    model: FznModel,
    solver: BacktrackSolver,
    out: TextIO,
    all_solutions: bool,
    num_solutions: Optional[int],
    output_mode: str = "item",
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
    """
    limit = None if all_solutions else (num_solutions if num_solutions is not None else 1)
    count = 0
    found = False
    exhausted = True
    for solution in solver.solve():
        print_solution(model, solution, out, output_mode)
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
