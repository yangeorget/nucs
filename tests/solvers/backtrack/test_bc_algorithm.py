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
from typing import Any

import pytest
from numba import int32, int64, njit, uint32  # type: ignore
from numpy.typing import NDArray

from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_ALLDIFFERENT,
    ALG_COUNT_EQ,
    ALG_LINEAR_EQ_C,
    ALG_LINEAR_GEQ_C,
    ALG_LINEAR_LEQ_C,
)
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.backtrack.bc_algorithm import bc_algorithm
from nucs.solvers.backtrack.consistency_algorithms import SIGN_CONSISTENCY_ALG
from nucs.solvers.buckets import buckets_add, buckets_pop
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
        """
        assert_compiled_without_refcounts(bc_algorithm, "bc_algorithm")


def assert_compiled_without_refcounts(
    consistency_algorithm: Callable, name: str, signature: Any = SIGN_CONSISTENCY_ALG
) -> None:
    """
    Checks that a propagation loop compiled without the reference-counting runtime holds no refcount call, whatever
    compiled the functions it calls.

    The compilation is a fresh one, since a dispatcher loaded from the cache has no IR. The queue helpers are
    first compiled in a refcounted context, with the loop's exact argument types: a helper that is not inlined
    would carry those refcounts into the loop, which on a warm cache happens whenever an ordinary caller
    compiled it first -- forcing it here keeps the test independent of the cache's history.

    :param consistency_algorithm: the jitted propagation loop
    :type consistency_algorithm: Callable
    :param name: the name of its function, to find it in the IR
    :type name: str
    :param signature: the signature to compile it with, defaults to that of the consistency algorithms
    :type signature: Any
    """

    @njit(int64(int32[::1], uint32[::1], int32, int64))
    def compile_queue_helpers_with_refcounting(
        buckets: NDArray, priorities: NDArray, idx: int, membership_offset: int
    ) -> int:
        buckets_add(buckets, priorities, idx, membership_offset)
        return buckets_pop(buckets, membership_offset)

    fresh = njit(_nrt=False)(consistency_algorithm.py_func)  # type: ignore[attr-defined, call-overload]
    fresh.compile(signature)
    ir = fresh.inspect_llvm(fresh.signatures[0])
    # the function itself, not the cpython and cfunc wrappers Numba also emits around it
    starts = [
        match.start()
        for match in re.finditer(rf"^define [^\n]*{name}[^\n]*$", ir, re.MULTILINE)
        if "cpython" not in match.group(0) and "cfunc" not in match.group(0)
    ]
    assert starts
    body = ir[starts[0] : ir.index("\n}\n", starts[0])]
    assert "@NRT_incref" not in body
    assert "@NRT_decref" not in body


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
        # 600 small problems in which a propagator has a variable twice: the solver finds exactly the solutions of an
        # enumeration. Before the write-back of copies (update_domains_of_copies), 27 of them failed: 17 with
        # linear_eq_c, 4 with linear_leq_c, 4 with linear_geq_c, 2 with count_eq. With that write-back but the watcher
        # of the first copy only (variable_trigger), 2 failed, with linear_eq_c: a zero coefficient on the first copy
        # watches nothing
        rng = random.Random(20261112)
        for _ in range(600):
            problem, bounds, is_solution = random_problem_with_copies(rng)
            expected = sorted(
                point for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]) if is_solution(point)
            )
            found = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all())
            assert found == expected, f"{bounds} {problem.propagators}"


def random_problem_with_copies(
    rng: random.Random,
) -> tuple[Problem, list[tuple[int, int]], Callable[[tuple[int, ...]], bool]]:
    """
    Returns a small problem with one propagator that has a variable more than once.

    The propagator is count_eq, linear_eq_c, linear_leq_c, linear_geq_c or alldifferent. The linear ones get zero and
    negative coefficients too, so that the watchers of the copies of a variable differ.

    :param rng: the source of the problem
    :type rng: random.Random

    :return: the problem, its bounds, and whether a point of the bounds is a solution
    :rtype: tuple[Problem, list[tuple[int, int]], Callable[[tuple[int, ...]], bool]]
    """
    bounds = random_bounds(rng, 3, 0, 3)
    problem = Problem(bounds)
    variables = [rng.randrange(3) for _ in range(3)]
    first, second = rng.sample(range(3), 2)
    variables[second] = variables[first]  # at least one copy
    kind = rng.randrange(5)
    if kind == 0:  # the number of v in the list is the last variable
        variables.append(rng.choice(variables))
        value = rng.randint(0, 3)
        problem.add_propagator(ALG_COUNT_EQ, variables, [value])
        return problem, bounds, lambda x: sum(x[v] == value for v in variables[:-1]) == x[variables[-1]]
    if kind == 4:
        problem.add_propagator(ALG_ALLDIFFERENT, variables)
        return problem, bounds, lambda x: len({x[v] for v in variables}) == len(variables)
    # sum a_i * x_i = c, <= c or >= c
    coefficients = [rng.choice([-2, -1, 0, 1, 2]) for _ in variables]
    constant = rng.randint(-3, 6)
    algorithm, holds = [
        (ALG_LINEAR_EQ_C, lambda total: total == constant),
        (ALG_LINEAR_LEQ_C, lambda total: total <= constant),
        (ALG_LINEAR_GEQ_C, lambda total: total >= constant),
    ][kind - 1]
    problem.add_propagator(algorithm, variables, coefficients + [constant])
    return problem, bounds, lambda x: holds(sum(a * x[v] for a, v in zip(coefficients, variables)))
