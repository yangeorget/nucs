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
import os
import shutil
import subprocess
import tempfile

from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run

SHARE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nucs", "fzn", "share")
MINIZINC = os.environ.get("MINIZINC") or shutil.which("minizinc")


def solve_fzn(
    fzn: str,
    all_solutions: bool = False,
    num_solutions: int | None = None,
    output_mode: str = "item",
    output_objective: bool = False,
    intermediate_solutions: bool = False,
    time_limit_ms: int | None = None,
    lcg: bool = False,
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
    :param lcg: whether an LcgSolver solves the model
    :type lcg: bool

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
        lcg=lcg,
    )
    return out.getvalue()


def compile_to_fzn(model: str, ozn_path: str | None = None) -> str:
    """
    Compiles a full MiniZinc model, with its own solve item, to FlatZinc with the NuCS library of globals.

    :param model: the MiniZinc text
    :type model: str
    :param ozn_path: the file that gets the output model, which prints the solutions as the model asks, or None
    :type ozn_path: Optional[str]

    :return: the FlatZinc text
    :rtype: str
    """
    assert MINIZINC is not None
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.mzn")
        with open(model_path, "w") as f:
            f.write(model)
        result = subprocess.run(
            [MINIZINC, "-c", "--solver", "nucs", model_path, "--output-fzn-to-stdout"]
            + ([] if ozn_path is None else ["--ozn", ozn_path]),
            capture_output=True,
            text=True,
            env=dict(os.environ, MZN_SOLVER_PATH=SHARE_DIR),
            timeout=120,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    return result.stdout
