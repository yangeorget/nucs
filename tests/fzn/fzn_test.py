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
Helpers shared by the FlatZinc tests.
"""

import io

from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run


def solve_fzn(
    fzn: str,
    all_solutions: bool = False,
    num_solutions: int | None = None,
    output_mode: str = "item",
    output_objective: bool = False,
    intermediate_solutions: bool = False,
    time_limit_ms: int | None = None,
) -> str:
    """
    Builds and solves a FlatZinc model, and returns the stream of solutions that the runner prints.

    :param fzn: the FlatZinc text
    :type fzn: str
    :param all_solutions: whether to print all the solutions
    :type all_solutions: bool
    :param num_solutions: the number of solutions to print, or None
    :type num_solutions: Optional[int]
    :param output_mode: the output mode, item, dzn or json
    :type output_mode: str
    :param output_objective: whether to print the objective with each solution
    :type output_objective: bool
    :param intermediate_solutions: whether to print each improving solution of an optimization
    :type intermediate_solutions: bool
    :param time_limit_ms: the time limit in milliseconds, or None
    :type time_limit_ms: Optional[int]

    :return: the printed stream
    :rtype: str
    """
    out = io.StringIO()
    run(
        build_model(parse(fzn)),
        out,
        all_solutions=all_solutions,
        num_solutions=num_solutions,
        output_mode=output_mode,
        output_objective=output_objective,
        intermediate_solutions=intermediate_solutions,
        time_limit_ms=time_limit_ms,
    )
    return out.getvalue()
