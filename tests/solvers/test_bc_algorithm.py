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
import itertools
import random
import re
from collections.abc import Callable

import pytest
from numba import int32, int64, njit, uint32  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_add, buckets_pop
from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_COUNT_EQ, ALG_LINEAR_EQ_C
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.bc_algorithm import bc_algorithm
from nucs.solvers.consistency_algorithms import SIGN_CONSISTENCY_ALG
from tests.propagators.propagator_test import random_bounds


@pytest.mark.skipif(bool(NUMBA_DISABLE_JIT), reason="inspects the compiled code, which only exists under the JIT")
class TestBcAlgorithmCompilation:
    def test_is_compiled_without_the_reference_counting_runtime(self) -> None:
        """
        Pins the private _nrt option on bc_algorithm, which a Numba upgrade or a tidy-up could otherwise drop.
        """
        assert bc_algorithm.targetoptions["_nrt"] is False

    def test_compiled_body_has_no_refcount_calls(self) -> None:
        """
        Checks that the compiled loop holds no refcount call, whatever compiled the functions it calls.

        The compilation is a fresh one, since a dispatcher loaded from the cache has no IR. The queue helpers are
        first compiled in a refcounted context, with the loop's exact argument types: a helper that is not inlined
        would carry those refcounts into the loop, which on a warm cache happens whenever an ordinary caller
        compiled it first -- forcing it here keeps the test independent of the cache's history.
        """

        @njit(int64(int32[::1], uint32[::1], int32, int64))
        def compile_queue_helpers_with_refcounting(
            buckets: NDArray, priorities: NDArray, idx: int, membership_offset: int
        ) -> int:
            buckets_add(buckets, priorities, idx, membership_offset)
            return buckets_pop(buckets, membership_offset)

        fresh = njit(_nrt=False)(bc_algorithm.py_func)  # type: ignore[call-overload]
        fresh.compile(SIGN_CONSISTENCY_ALG)
        ir = fresh.inspect_llvm(fresh.signatures[0])
        # the function itself, not the cpython and cfunc wrappers Numba also emits around it
        starts = [
            match.start()
            for match in re.finditer(r"^define [^\n]*bc_algorithm[^\n]*$", ir, re.MULTILINE)
            if "cpython" not in match.group(0) and "cfunc" not in match.group(0)
        ]
        assert starts
        body = ir[starts[0] : ir.index("\n}\n", starts[0])]
        assert "@NRT_incref" not in body
        assert "@NRT_decref" not in body


def random_problem_with_copies(
    rng: random.Random,
) -> tuple[Problem, list[tuple[int, int]], Callable[[tuple[int, ...]], bool]]:
    """
    Returns a small problem with one propagator that has a variable more than once, its bounds and its solution
    predicate.
    """
    bounds = random_bounds(rng, 3, 0, 3)
    problem = Problem(bounds)
    kind = rng.randrange(3)
    variables = [rng.randrange(3) for _ in range(3)]
    variables[rng.randrange(3)] = variables[rng.randrange(3)] = rng.randrange(3)  # at least one copy, mostly
    if kind == 0:  # the number of v in the list is the last variable
        variables.append(rng.choice(variables))
        value = rng.randint(0, 3)
        problem.add_propagator(ALG_COUNT_EQ, variables, [value])
        return problem, bounds, lambda x: sum(x[v] == value for v in variables[:-1]) == x[variables[-1]]
    if kind == 1:  # sum a_i * x_i = c
        coefficients = [rng.choice([-2, -1, 1, 2]) for _ in variables]
        constant = rng.randint(-3, 6)
        problem.add_propagator(ALG_LINEAR_EQ_C, variables, coefficients + [constant])
        return problem, bounds, lambda x: sum(a * x[v] for a, v in zip(coefficients, variables)) == constant
    problem.add_propagator(ALG_ALLDIFFERENT, variables)
    return problem, bounds, lambda x: len({x[v] for v in variables}) == len(variables)


class TestBcAlgorithmCopies:
    @pytest.mark.parametrize(
        "bounds,variables,value",
        [
            # x1 is the number of 1 in [x0, x1, x2]: the list copy of x1 gets 0 and the count copy gets 1
            ([(0, 0), (0, 1), (1, 1)], [0, 1, 2, 1], 1),
            # x4 is the number of 3 in [x0, x1, x2, x3]: the copies of x4 disagree, and the last one undid the first
            ([(3, 3), (3, 3), (3, 3), (1, 2), (1, 3)], [0, 1, 2, 3, 4, 4], 3),
        ],
    )
    def test_find_all_copies_with_no_common_value_have_no_solution(
        self, bounds: list[tuple[int, int]], variables: list[int], value: int
    ) -> None:
        # before the write-back of copies, both gave a wrong solution: [0, 0, 1], and [3, 3, 3, 1, 3], [3, 3, 3, 2, 3]
        problem = Problem(bounds)
        problem.add_propagator(ALG_COUNT_EQ, variables, [value])
        assert list(BacktrackSolver(problem).find_all()) == []

    def test_find_all_soundness_against_brute_force(self) -> None:
        # 300 small problems in which a propagator has a variable twice: the solver finds exactly the solutions of an
        # enumeration; before the write-back of copies, 14 of them failed: 10 with linear_eq_c, 4 with count_eq
        rng = random.Random(20261112)
        for _ in range(300):
            problem, bounds, is_solution = random_problem_with_copies(rng)
            expected = sorted(
                point for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]) if is_solution(point)
            )
            found = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all())
            assert found == expected, f"{bounds} {problem.propagators}"
