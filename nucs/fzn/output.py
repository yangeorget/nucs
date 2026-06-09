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

from nucs.fzn.parser import Id

if TYPE_CHECKING:
    from nucs.fzn.model import FznModel

SOLUTION_SEPARATOR = "----------"
SEARCH_COMPLETE = "=========="
UNSATISFIABLE = "=====UNSATISFIABLE====="


def print_solution(model: "FznModel", solution: NDArray, out: TextIO) -> None:
    """
    Prints a single solution as ``name = value;`` / ``name = array1d(lo..hi, [..]);`` items followed by the
    solution separator.

    :param model: the model holding the output items
    :type model: FznModel
    :param solution: the solution array indexed by NuCS variable
    :type solution: NDArray
    :param out: the output stream
    :type out: TextIO
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
    out.write(SOLUTION_SEPARATOR + "\n")


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
