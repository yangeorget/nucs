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
Formats NuCS solutions as the FlatZinc solution stream that MiniZinc's ``solns2out``/``.ozn`` consumes.
"""

from typing import TYPE_CHECKING, TextIO

from numpy.typing import NDArray

if TYPE_CHECKING:
    from nucs.fzn.model import FznModel

SOLUTION_SEPARATOR = "----------"
SEARCH_COMPLETE = "=========="
UNSATISFIABLE = "=====UNSATISFIABLE====="
UNKNOWN = "=====UNKNOWN====="

OUTPUT_OBJECTIVE_NAME = "_objective"


def print_solution(
    model: "FznModel",
    solution: NDArray,
    out: TextIO,
    output_mode: str = "item",
    objective_value: int | None = None,
) -> None:
    """
    Prints a single solution followed by the solution separator.

    In ``item``/``dzn`` mode each output variable is printed as ``name = value;`` /
    ``name = array1d(lo..hi, [..]);``; in ``json`` mode the solution is printed as a JSON object. When
    ``objective_value`` is given it is appended under the ``_objective`` name.

    :param model: the model holding the output items
    :type model: FznModel
    :param solution: the solution array indexed by NuCS variable
    :type solution: NDArray
    :param out: the output stream
    :type out: TextIO
    :param output_mode: the output format, one of ``item``, ``dzn`` or ``json``
    :type output_mode: str
    :param objective_value: the objective value to print, or None to omit it
    :type objective_value: Optional[int]
    """
    if output_mode == "json":
        _print_solution_json(model, solution, out, objective_value)
    else:
        _print_solution_dzn(model, solution, out, objective_value)
    out.write(SOLUTION_SEPARATOR + "\n")


def _print_solution_dzn(model: "FznModel", solution: NDArray, out: TextIO, objective_value: int | None) -> None:
    """
    Prints a solution as a FlatZinc assignment stream (the ``item``/``dzn`` output mode).

    :param model: the model holding the output items
    :type model: FznModel
    :param solution: the solution array indexed by NuCS variable
    :type solution: NDArray
    :param out: the output stream
    :type out: TextIO
    :param objective_value: the objective value to print, or None to omit it
    :type objective_value: Optional[int]
    """
    plan = model.output_plan
    values = plan.values(solution)
    lines = []
    for item in plan.items:
        if item.is_array:
            body = _join(values[item.start : item.end], item.is_bool)
            lines.append(f"{item.name} = array1d({item.lo}..{item.hi}, [{body}]);\n")
        else:
            lines.append(f"{item.name} = {_fmt(values[item.start], item.is_bool)};\n")
    if objective_value is not None:
        lines.append(f"{OUTPUT_OBJECTIVE_NAME} = {objective_value};\n")
    out.write("".join(lines))


def _print_solution_json(model: "FznModel", solution: NDArray, out: TextIO, objective_value: int | None) -> None:
    """
    Prints a solution as a JSON object (the ``json`` output mode).

    :param model: the model holding the output items
    :type model: FznModel
    :param solution: the solution array indexed by NuCS variable
    :type solution: NDArray
    :param out: the output stream
    :type out: TextIO
    :param objective_value: the objective value to print, or None to omit it
    :type objective_value: Optional[int]
    """
    plan = model.output_plan
    values = plan.values(solution)
    entries = []
    for item in plan.items:
        if item.is_array:
            entries.append(f'  "{item.name}" : [{_join(values[item.start : item.end], item.is_bool)}]')
        else:
            entries.append(f'  "{item.name}" : {_fmt(values[item.start], item.is_bool)}')
    if objective_value is not None:
        entries.append(f'  "{OUTPUT_OBJECTIVE_NAME}" : {objective_value}')
    out.write("{\n" + ",\n".join(entries) + "\n}\n")


def _fmt(value: int, is_bool: bool) -> str:
    """
    Formats a value for the FlatZinc solution stream: booleans as ``true``/``false``, integers as digits.

    :param value: the value
    :type value: int
    :param is_bool: whether the value belongs to a boolean variable
    :type is_bool: bool

    :return: the formatted value
    :rtype: str
    """
    if is_bool:
        return "true" if value else "false"
    return str(value)


def _join(values: list[int], is_bool: bool) -> str:
    """
    Formats the values of an array for the FlatZinc solution stream, separated by commas.

    :param values: the values
    :type values: list[int]
    :param is_bool: whether the values belong to boolean variables
    :type is_bool: bool

    :return: the formatted values
    :rtype: str
    """
    if is_bool:
        return ", ".join(["true" if value else "false" for value in values])
    return ", ".join(map(str, values))


def print_search_complete(out: TextIO) -> None:
    """
    Prints the search-complete marker (the whole space was explored or the optimum was proven).

    :param out: the output stream
    :type out: TextIO
    """
    out.write(SEARCH_COMPLETE + "\n")


def print_unknown(out: TextIO) -> None:
    """
    Prints the unknown marker: no solution was found, but the search space was not fully explored.

    :param out: the output stream
    :type out: TextIO
    """
    out.write(UNKNOWN + "\n")


def print_unsatisfiable(out: TextIO) -> None:
    """
    Prints the unsatisfiable marker.

    :param out: the output stream
    :type out: TextIO
    """
    out.write(UNSATISFIABLE + "\n")
