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

from typing import TYPE_CHECKING, Optional, TextIO

from numpy.typing import NDArray

from nucs.fzn.parser import Id

if TYPE_CHECKING:
    from nucs.fzn.model import FznModel

SOLUTION_SEPARATOR = "----------"
SEARCH_COMPLETE = "=========="
UNSATISFIABLE = "=====UNSATISFIABLE====="

OUTPUT_OBJECTIVE_NAME = "_objective"


def print_solution(
    model: "FznModel",
    solution: NDArray,
    out: TextIO,
    output_mode: str = "item",
    objective_value: Optional[int] = None,
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


def _print_solution_dzn(
    model: "FznModel", solution: NDArray, out: TextIO, objective_value: Optional[int]
) -> None:
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
    for item in model.output_items:
        if item[0] == "scalar":
            _, name, is_bool = item
            out.write(f"{name} = {_fmt(model.value_of(_id(name), solution), is_bool)};\n")
        else:
            _, name, lo, hi, is_bool = item
            values = [model.value_of(e, solution) for e in model.elements_of(_id(name))]
            body = ", ".join(_fmt(v, is_bool) for v in values)
            out.write(f"{name} = array1d({lo}..{hi}, [{body}]);\n")
    if objective_value is not None:
        out.write(f"{OUTPUT_OBJECTIVE_NAME} = {objective_value};\n")


def _print_solution_json(
    model: "FznModel", solution: NDArray, out: TextIO, objective_value: Optional[int]
) -> None:
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
    entries = []
    for item in model.output_items:
        if item[0] == "scalar":
            _, name, is_bool = item
            entries.append(f'  "{name}" : {_json_fmt(model.value_of(_id(name), solution), is_bool)}')
        else:
            _, name, lo, hi, is_bool = item
            values = [model.value_of(e, solution) for e in model.elements_of(_id(name))]
            body = ", ".join(_json_fmt(v, is_bool) for v in values)
            entries.append(f'  "{name}" : [{body}]')
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


def _json_fmt(value: int, is_bool: bool) -> str:
    """
    Formats a value for the JSON solution stream: booleans as ``true``/``false``, integers as digits.

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


def print_search_complete(out: TextIO) -> None:
    """
    Prints the search-complete marker (the whole space was explored or the optimum was proven).

    :param out: the output stream
    :type out: TextIO
    """
    out.write(SEARCH_COMPLETE + "\n")


def print_unsatisfiable(out: TextIO) -> None:
    """
    Prints the unsatisfiable marker.

    :param out: the output stream
    :type out: TextIO
    """
    out.write(UNSATISFIABLE + "\n")


def _id(name: str):  # type: ignore[no-untyped-def]
    """
    Wraps an identifier name in an :class:`Id` term.

    :param name: the identifier name
    :type name: str

    :return: an Id term
    :rtype: Id
    """
    return Id(name)
