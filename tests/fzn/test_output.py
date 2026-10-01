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
import io

import numpy as np
import pytest

from nucs.fzn.model import build_model
from nucs.fzn.output import print_solution
from nucs.fzn.parser import parse


class TestOutput:
    @pytest.mark.parametrize(
        "fzn,values,output_mode,expected",
        [
            # no output term is a variable: the solution has no value at all, and the plan must not read it
            (
                (
                    "var 0..10: k :: output_var = 7;\n"
                    "array [1..2] of var int: c :: output_array([1..2]) = [4, true];\nsolve satisfy;"
                ),
                {},
                "item",
                "k = 7;\nc = array1d(1..2, [4, 1]);\n----------\n",
            ),
            # accesses and constants in one array; the constant -1 is not the -1 that marks a constant term in the plan
            (
                (
                    "array [1..2] of var 0..9: x :: output_array([1..2]);\n"
                    "array [1..4] of var int: m :: output_array([0..1, 1..2]) = [x[2], -1, x[1], 3];\nsolve satisfy;"
                ),
                {"x[1]": 5, "x[2]": 6},
                "item",
                "x = array1d(1..2, [5, 6]);\nm = array1d(0..1, [6, -1, 5, 3]);\n----------\n",
            ),
            # booleans as true/false, an alias, and an empty array
            (
                (
                    "var bool: a :: output_var;\nvar bool: b;\nvar 0..3: y :: output_var;\nvar 0..3: z :: output_var = y;\n"
                    "array [1..2] of var bool: bs :: output_array([1..2]) = [a, b];\n"
                    "array [1..0] of var int: e :: output_array([1..0]) = [];\nsolve satisfy;"
                ),
                {"a": 1, "b": 0, "y": 2},
                "item",
                "a = true;\ny = 2;\nz = 2;\nbs = array1d(1..2, [true, false]);\ne = array1d(1..0, []);\n----------\n",
            ),
            (
                "var bool: a :: output_var;\narray [1..3] of var 0..9: x :: output_array([1..3]);\nsolve satisfy;",
                {"a": 0, "x[1]": 1, "x[2]": 2, "x[3]": 3},
                "json",
                '{\n  "a" : false,\n  "x" : [1, 2, 3]\n}\n----------\n',
            ),
        ],
    )
    def test_print_solution(self, fzn: str, values: dict[str, int], output_mode: str, expected: str) -> None:
        model = build_model(parse(fzn))
        solution = np.zeros(len(model.problem.domains), dtype=np.int32)
        for name, value in values.items():
            solution[model.vars[name]] = value
        out = io.StringIO()
        print_solution(model, solution, out, output_mode)
        assert out.getvalue() == expected
