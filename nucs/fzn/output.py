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
            name = item[1]
            out.write(f"{name} = {model.value_of(_id(name), solution)};\n")
        else:
            _, name, lo, hi = item
            values = [model.value_of(e, solution) for e in model.elements_of(_id(name))]
            body = ", ".join(str(v) for v in values)
            out.write(f"{name} = array1d({lo}..{hi}, [{body}]);\n")
    out.write(SOLUTION_SEPARATOR + "\n")


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
