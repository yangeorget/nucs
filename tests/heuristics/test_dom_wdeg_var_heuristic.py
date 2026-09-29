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
import random
from collections.abc import Sequence

from nucs.heuristics.dom_wdeg_var_heuristic import dom_wdeg_var_heuristic
from nucs.propagators.propagators import ALG_DUMMY, ALG_LINEAR_LEQ_C
from tests.heuristics.var_heuristic_test import call_var_heuristic


def _reference(
    domains: Sequence[tuple[int, int]],
    propagators: Sequence[Sequence[int]],
    entailed: Sequence[int],
    weights: Sequence[float],
) -> int:
    """The definition, written as plainly as possible: the oracle for the jitted heuristic."""
    best: tuple[bool, float] | None = None
    best_variable = -1
    for variable, (lo, hi) in enumerate(domains):
        if lo == hi:
            continue
        wdeg = 0.0
        for propagator, variables in enumerate(propagators):
            unbound = {v for v in variables if domains[v][0] < domains[v][1]}
            if variable in unbound and not entailed[propagator] and len(unbound) >= 2:
                wdeg += weights[propagator]
        key = (wdeg == 0.0, (hi - lo + 1) if wdeg == 0.0 else (hi - lo + 1) / wdeg)
        if best is None or key < best:
            best, best_variable = key, variable
    return best_variable


class TestDomWdegVarHeuristic:
    def test_selects_smallest_ratio_of_domain_size_to_weighted_degree(self) -> None:
        domains = [(0, 3), (0, 3), (0, 1)]
        propagators = [[0, 1], [1, 2]]
        # weights 1: ratios 4/1, 4/2 and 2/1, the tie goes to x1
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=propagators) == 1
        # p1 failed 5 times: ratios 4/1, 4/7 and 2/6
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=propagators, weights=[1, 6]) == 2
        # p0 failed 5 times: ratios 4/6, 4/7 and 2/1
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=propagators, weights=[6, 1]) == 1

    def test_ignores_an_entailed_propagator(self) -> None:
        # without p0, x0 has no live propagator and comes last although its domain is the smallest
        domains = [(0, 1), (0, 5), (0, 5), (0, 5)]
        propagators = [[0, 1], [2, 3]]
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=propagators) == 0
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=propagators, entailed=[1, 0]) == 2

    def test_ignores_a_propagator_with_a_single_unbound_variable(self) -> None:
        # x1 is bound, so p0 cannot fail on a decision about x0 alone
        domains = [(0, 1), (4, 4), (0, 5), (0, 5)]
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=[[0, 1], [2, 3]], weights=[9, 1]) == 2

    def test_counts_distinct_unbound_variables(self) -> None:
        # p0 lists x0 twice: it still has a single unbound variable, and is not live
        domains = [(0, 1), (4, 4), (0, 5), (0, 5)]
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, propagators=[[0, 1, 0], [2, 3]]) == 2

    def test_ignores_a_propagator_that_does_not_watch_the_variable(self) -> None:
        # 0 * x0 + x1 <= 5 failed 9 times, but it cannot fail because of x0: x0 has no weighted degree and comes last,
        # and x1 (weights 10 + 1) wins over x2 (weight 1)
        domains = [(0, 1), (0, 5), (0, 5)]
        propagators = [[0, 1], [1, 2]]
        algorithms = [(ALG_LINEAR_LEQ_C, [0, 1, 5]), (ALG_DUMMY, [])]
        chosen = call_var_heuristic(
            dom_wdeg_var_heuristic, domains, propagators=propagators, weights=[10, 1], algorithms=algorithms
        )
        assert chosen == 1

    def test_selects_a_variable_without_live_propagator_last(self) -> None:
        # x0 has no propagator, x2 a live one although x1 is not a decision variable
        domains = [(0, 1), (0, 9), (0, 5)]
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, [0, 2], propagators=[[1, 2]]) == 2
        assert call_var_heuristic(dom_wdeg_var_heuristic, domains, [0], propagators=[[1, 2]]) == 0

    def test_returns_minus_one_when_all_instantiated(self) -> None:
        assert call_var_heuristic(dom_wdeg_var_heuristic, [(3, 3), (7, 7)], propagators=[[0, 1]]) == -1

    def test_agrees_with_the_definition(self) -> None:
        rng = random.Random(20260925)
        for _ in range(300):
            variable_nb = rng.randint(1, 6)
            domains = []
            for _ in range(variable_nb):
                lo = rng.randint(0, 3)
                domains.append((lo, lo + rng.choice([0, 0, 1, 2, 3])))
            propagators = [
                [rng.randrange(variable_nb) for _ in range(rng.randint(1, 4))] for _ in range(rng.randint(0, 5))
            ]
            entailed = [rng.choice([0, 0, 1]) for _ in propagators]
            weights = [float(rng.randint(1, 4)) for _ in propagators]
            expected = _reference(domains, propagators, entailed, weights)
            actual = call_var_heuristic(
                dom_wdeg_var_heuristic, domains, propagators=propagators, entailed=entailed, weights=weights
            )
            assert actual == expected, f"{domains} {propagators} {entailed} {weights}"
