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
from collections.abc import Callable

import numpy as np
import pytest
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.numba_helper import addresses_from_functions
from nucs.problems.problem import Problem
from nucs.propagators.explanations import EXPLAIN_SIDE, EXPLAIN_VALUE, EXPLAIN_VAR_IDX, explain_naive
from nucs.propagators.neq_propagator import compute_domains_neq, get_complexity_neq, get_triggers_neq
from nucs.propagators.propagators import (
    ALG_COUNT_EQ,
    ALG_DUMMY,
    ALG_MUL_EQ,
    ALG_NEQ,
    COMPUTE_DOMAINS_FCTS,
    EXPLAIN_FCTS,
    GET_STATE_FCTS,
    SIGN_COMPUTE_DOMAINS,
    register_propagator,
)
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.explanations import (
    EXPLANATION_CONFIRMED,
    EXPLANATION_ENUMERATED,
    EXPLANATION_SUSPICIOUS,
    EXPLANATION_WRONG,
    bound_at,
    check_explanation,
    explanation_literals,
    first_copies,
)
from nucs.solvers.lcg.implication_trail import ITRAIL_WIDTH, REASON_PROPAGATOR, itrail_push
from nucs.solvers.lcg.lcg_solver import LcgSolver


@njit(cache=True)
def explain_nothing(domains: NDArray, parameters: NDArray, var_idx: int, side: int, value: int, out: NDArray) -> int:
    # wrong for each literal that the propagator sets: the root box does not force it
    return 0


@njit(cache=True)
def explain_failures_with_nothing(
    domains: NDArray, parameters: NDArray, var_idx: int, side: int, value: int, out: NDArray
) -> int:
    # right for each literal, wrong for each failure
    if var_idx < 0:
        return 0
    return explain_naive(domains, parameters, var_idx, side, value, out)


@njit(cache=True)
def explain_with_a_later_literal(
    domains: NDArray, parameters: NDArray, var_idx: int, side: int, value: int, out: NDArray
) -> int:
    # [x >= min + 1] on the first variable: sufficient with the rest of the box, but not true when the call read it
    row_nb = explain_naive(domains, parameters, var_idx, side, value, out)
    out[0, EXPLAIN_VALUE] = domains[0, DOMAIN_MIN] + 1
    return row_nb


def neq_triangle(explain_fct: Callable) -> Problem:
    # three variables in [0, 1] that differ pairwise: no solution, so the search fails, and each decision makes the
    # neq propagators set literals
    alg = register_propagator(get_triggers_neq, get_complexity_neq, compute_domains_neq, explain_fct=explain_fct)
    problem = Problem([(0, 1)] * 3)
    for x, y in [(0, 1), (0, 2), (1, 2)]:
        problem.add_propagator(alg, [x, y])
    return problem


class TestExplanations:
    @pytest.mark.parametrize(
        "var_side,pos,expected",
        [
            (0, 0, 0),  # no row below pos: the root value
            (0, 1, 2),
            (0, 3, 4),  # row 3 is not below pos 3
            (0, 4, 5),  # the last row of the chain
            (3, 1, 9),  # x1 MAX changes at row 1, which is not below pos 1
            (3, 2, 7),
            (1, 4, 9),  # a bound with no row
        ],
    )
    def test_bound_at(self, var_side: int, pos: int, expected: int) -> None:
        initial_domains = np.array([[0, 9], [0, 9]], dtype=np.int32)
        itrail = np.zeros((4, ITRAIL_WIDTH), dtype=np.int32)
        itrail_last = np.full(4, -1, dtype=np.int32)
        itrail_sz = 0
        for row_var_side, value in [(0, 2), (3, 7), (0, 4), (0, 5)]:
            itrail_sz = itrail_push(itrail, itrail_last, itrail_sz, row_var_side, value, 0, REASON_PROPAGATOR, 0, 0)
        assert bound_at(itrail, itrail_last, initial_domains, var_side, pos) == expected

    def test_explanation_literals_maps_the_variables_and_drops_the_root_literals(self) -> None:
        initial_domains = np.array([[0, 9]] * 6, dtype=np.int32)
        propagator_variables = np.array([7, 5, 2], dtype=np.int32)  # the propagator starts at index 1
        explanation = np.array(
            [
                [0, DOMAIN_MIN, 0],  # [x5 >= 0] is true at the root
                [0, DOMAIN_MAX, 3],
                [1, DOMAIN_MIN, 1],
                [1, DOMAIN_MAX, 9],  # [x2 <= 9] is true at the root
            ],
            dtype=np.int32,
        )
        literals = np.zeros((4, 2), dtype=np.int32)
        lit_nb = explanation_literals(explanation, 4, propagator_variables, 1, initial_domains, literals)
        assert literals[:lit_nb].tolist() == [[(5 << 1) | DOMAIN_MAX, 3], [2 << 1, 1]]

    @pytest.mark.parametrize(
        "algorithm,parameters,variables,box,var_idx,side,value,expected",
        [
            # the propagation fails
            (ALG_NEQ, [], [0, 1], [(1, 1), (1, 1)], -1, 0, 0, EXPLANATION_CONFIRMED),
            # the propagation sets the literal
            (ALG_NEQ, [], [0, 1], [(1, 1), (1, 2)], 1, DOMAIN_MIN, 2, EXPLANATION_CONFIRMED),
            # x * y = 1 with x, y in [-3, 0] forces x = y = -1, but the propagation leaves the box as it is
            (ALG_MUL_EQ, [], [0, 1, 2], [(-3, 0), (-3, 0), (1, 1)], 0, DOMAIN_MAX, -1, EXPLANATION_ENUMERATED),
            # x = 0, y = 1 satisfies x != y and not [x >= 1]
            (ALG_NEQ, [], [0, 1], [(0, 2), (0, 2)], 0, DOMAIN_MIN, 1, EXPLANATION_WRONG),
            # x = 0, y = 1 satisfies x != y: the box has a solution
            (ALG_NEQ, [], [0, 1], [(0, 2), (0, 2)], -1, 0, 0, EXPLANATION_WRONG),
            # x != x has no solution, but only the tuples with two equal copies show it: (0, 1) is not a tuple
            (ALG_NEQ, [], [0, 0], [(0, 2), (0, 2)], -1, 0, 0, EXPLANATION_ENUMERATED),
            # x1 is the number of 1 in [x0, x1]: with x0 = 1, the propagation sets [x1 >= 1] on the count, which is the
            # second copy of x1, and the engine writes it on both copies
            (ALG_COUNT_EQ, [1], [0, 1, 1], [(1, 1), (0, 3), (0, 3)], 1, DOMAIN_MIN, 1, EXPLANATION_CONFIRMED),
            # the same literal on a box of a million tuples
            (ALG_NEQ, [], [0, 1], [(0, 999), (0, 999)], 0, DOMAIN_MIN, 1, EXPLANATION_SUSPICIOUS),
        ],
    )
    def test_check_explanation(
        self,
        algorithm: int,
        parameters: list[int],
        variables: list[int],
        box: list[tuple[int, int]],
        var_idx: int,
        side: int,
        value: int,
        expected: int,
    ) -> None:
        compute_domains_addrs = addresses_from_functions(
            COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS, np.array([algorithm]), ALG_DUMMY
        )
        verdict = check_explanation(
            compute_domains_addrs,
            algorithm,
            np.array(parameters, dtype=np.int32),
            sum(GET_STATE_FCTS[algorithm](len(variables), parameters)),
            first_copies(np.array(variables, dtype=np.int32)),
            np.array(box, dtype=np.int32),
            var_idx,
            side,
            value,
        )
        assert verdict == expected

    def test_explain_naive_is_the_default_explanation(self) -> None:
        assert EXPLAIN_FCTS[ALG_NEQ] is explain_naive
        out = np.zeros((4, 3), dtype=np.int32)
        domains = np.array([[1, 3], [2, 2]], dtype=np.int32)
        assert explain_naive(domains, np.zeros(0, dtype=np.int32), 0, DOMAIN_MIN, 2, out) == 4
        assert out[:, [EXPLAIN_VAR_IDX, EXPLAIN_SIDE, EXPLAIN_VALUE]].tolist() == [
            [0, DOMAIN_MIN, 1],
            [0, DOMAIN_MAX, 3],
            [1, DOMAIN_MIN, 2],
            [1, DOMAIN_MAX, 2],
        ]

    def test_debug_mode_accepts_a_correct_explain_function(self) -> None:
        # the control of the three tests below: the same problem, with the naive explanation given explicitly
        assert list(LcgSolver(neq_triangle(explain_naive), debug=True).solve()) == []

    @pytest.mark.parametrize(
        "explain_fct,message",
        [
            (explain_nothing, r"The explanation of row \d+ of \d+ is wrong"),
            (explain_failures_with_nothing, r"The explanation of the failure of propagator \d+ is wrong"),
            (explain_with_a_later_literal, r"The explanation of row \d+ of \d+ is not true"),
        ],
    )
    def test_debug_mode_finds_a_wrong_explanation(self, explain_fct: Callable, message: str) -> None:
        with pytest.raises(AssertionError, match=message):
            list(LcgSolver(neq_triangle(explain_fct), debug=True).solve())

    def test_find_all_checks_the_explanations_of_the_tier_b_propagators(self) -> None:
        # count_eq keeps a live set in its trailed prefix from LIVE_SET_MIN_ARITY = 16 variables (design/proposals/lcg.md,
        # Part 2): the debug mode calls it on the box of each explanation with a new state block at zero
        problem = MagicSequenceProblem(16)
        solutions = [solution.tolist() for solution in LcgSolver(problem, debug=True).solve()]
        assert solutions == [solution.tolist() for solution in BacktrackSolver(problem).solve()]
        assert len(solutions) == 1
