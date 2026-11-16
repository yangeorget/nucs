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
import runpy
import sys
from collections.abc import Callable

import pytest

from nucs.examples import default_argument_parser
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver


@pytest.fixture
def run_example_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[..., BacktrackSolver]:
    """
    Returns a function that runs the __main__ of an example as the command line does, and returns its solver.

    A test of the solver that the example builds checks the example itself: a test that builds its own solver would
    still pass after a change to the example's search, which is what this protects.

    :param monkeypatch: the pytest fixture that replaces sys.argv and the runners of the examples
    :type monkeypatch: pytest.MonkeyPatch
    :param capsys: the pytest fixture that captures what the example prints
    :type capsys: pytest.CaptureFixture[str]

    :return: a function of the example module and its arguments, which returns the solver once it has run
    :rtype: Callable[..., BacktrackSolver]
    """
    solvers: list[BacktrackSolver] = []
    for name in ("run_solver", "run_optimizer"):
        original = getattr(default_argument_parser, name)

        def recording(
            solver: BacktrackSolver, *args: object, _original: Callable = original, **kwargs: object
        ) -> object:
            solvers.append(solver)
            return _original(solver, *args, **kwargs)

        # the __main__ imports the runners by name when it runs, so it gets the recording ones
        monkeypatch.setattr(default_argument_parser, name, recording)

    def run(module: str, *argv: str) -> BacktrackSolver:
        monkeypatch.setattr(sys, "argv", [module, *argv, "--no-display-solutions", "--no-display-stats"])
        runpy.run_module(module, run_name="__main__")
        capsys.readouterr()
        return solvers[-1]

    return run
