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
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.eq_c_imp_propagator import compute_domains_eq_c_imp
from nucs.propagators.eq_imp_propagator import compute_domains_eq_imp
from nucs.propagators.leq_c_imp_propagator import compute_domains_leq_c_imp
from nucs.propagators.member_imp_propagator import compute_domains_member_imp
from nucs.propagators.neq_c_imp_propagator import compute_domains_neq_c_imp
from nucs.propagators.neq_imp_propagator import compute_domains_neq_imp
from tests.propagators.propagator_test import PropagatorTest


# reference semantics of the half-reified constraint r -> C, C over the operands (after the boolean r)
def _pred_eq_c(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] == params[0]


def _pred_eq(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] == ops[1]


def _pred_leq_c(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] <= ops[1] + params[0]


def _pred_neq(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] != ops[1]


def _pred_neq_c(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] != params[0]


def _pred_member(r: int, ops: list[int], params: list[int]) -> bool:
    return r == 0 or ops[0] in params


class TestImp(PropagatorTest):
    @pytest.mark.parametrize(
        "compute_fn,pred,n_ops,params",
        [
            (compute_domains_eq_c_imp, _pred_eq_c, 1, [2]),
            (compute_domains_eq_imp, _pred_eq, 2, []),
            (compute_domains_leq_c_imp, _pred_leq_c, 2, [0]),
            (compute_domains_leq_c_imp, _pred_leq_c, 2, [1]),
            (compute_domains_neq_imp, _pred_neq, 2, []),
            (compute_domains_neq_c_imp, _pred_neq_c, 1, [0]),
            (compute_domains_neq_c_imp, _pred_neq_c, 1, [2]),
            (compute_domains_member_imp, _pred_member, 1, [-2, 0, 2]),
            (compute_domains_member_imp, _pred_member, 1, [-1, 0, 1]),
            (compute_domains_member_imp, _pred_member, 1, [-1, 2]),
            (compute_domains_member_imp, _pred_member, 1, [0]),
        ],
    )
    def test_bound_consistency_against_brute_force(
        self, compute_fn: Callable, pred: Callable, n_ops: int, params: list[int]
    ) -> None:
        # the half-reified propagator must compute the exact bound-consistent projection of r -> C
        rng = range(-2, 3)
        op_boxes = [(lo, hi) for lo in rng for hi in rng if lo <= hi]
        p = np.array(params, dtype=np.int32)
        for b_dom in [(0, 0), (1, 1), (0, 1)]:
            for op_dom in itertools.product(op_boxes, repeat=n_ops):
                feasible = []
                for r in range(b_dom[0], b_dom[1] + 1):
                    for ops in itertools.product(*[range(lo, hi + 1) for lo, hi in op_dom]):
                        if pred(r, ops, params):
                            feasible.append((r, *ops))
                domains = np.array([list(b_dom), *[list(d) for d in op_dom]], dtype=np.int32)
                status = compute_fn(domains, p, np.empty(0, dtype=np.int32))
                if not feasible:
                    assert status == PROP_INCONSISTENCY, f"expected inconsistency b={b_dom} ops={op_dom} p={params}"
                    continue
                assert status in (PROP_CONSISTENCY, PROP_ENTAILMENT)
                for i in range(n_ops + 1):
                    vals = [t[i] for t in feasible]
                    assert domains[i, DOMAIN_MIN] == min(vals) and domains[i, DOMAIN_MAX] == max(vals), (
                        f"var {i} for b={b_dom} ops={op_dom} p={params}: got {list(domains[i])} exp [{min(vals)},{max(vals)}]"
                    )
                # idempotence: a second call on the projection changes nothing
                again = domains.copy()
                compute_fn(again, p, np.empty(0, dtype=np.int32))
                assert np.array_equal(again, domains), f"not idempotent for b={b_dom} ops={op_dom} p={params}"

    @pytest.mark.parametrize(
        "compute_fn,domains,parameters,consistency_result,expected_domains",
        [
            # r=1 forces x=c
            (compute_domains_eq_c_imp, [(1, 1), (0, 9)], [4], PROP_ENTAILMENT, [[1, 1], [4, 4]]),
            # r free, x=c impossible -> r=0
            (compute_domains_eq_c_imp, [(0, 1), (5, 9)], [4], PROP_ENTAILMENT, [[0, 0], [5, 9]]),
            # r free, x can still equal c -> no pruning (unlike full reif, x is NOT split)
            (compute_domains_eq_c_imp, [(0, 1), (0, 9)], [4], PROP_CONSISTENCY, [[0, 1], [0, 9]]),
            # r=0 -> vacuous, nothing enforced (x keeps c in its domain)
            (compute_domains_eq_c_imp, [(0, 0), (0, 9)], [4], PROP_ENTAILMENT, [[0, 0], [0, 9]]),
            # r=1 forces x=y intersection
            (compute_domains_eq_imp, [(1, 1), (0, 5), (3, 9)], [], PROP_CONSISTENCY, [[1, 1], [3, 5], [3, 5]]),
            # r free, x and y disjoint -> r=0
            (compute_domains_eq_imp, [(0, 1), (0, 2), (5, 9)], [], PROP_ENTAILMENT, [[0, 0], [0, 2], [5, 9]]),
            # r=1 forces x<=y
            (compute_domains_leq_c_imp, [(1, 1), (0, 9), (0, 4)], [0], PROP_CONSISTENCY, [[1, 1], [0, 4], [0, 4]]),
            # r free, x always > y -> r=0
            (compute_domains_leq_c_imp, [(0, 1), (7, 9), (0, 4)], [0], PROP_ENTAILMENT, [[0, 0], [7, 9], [0, 4]]),
            # r=1 forces x!=y where x is fixed; y loses 5, becoming disjoint from x -> entailed
            (compute_domains_neq_imp, [(1, 1), (5, 5), (3, 5)], [], PROP_ENTAILMENT, [[1, 1], [5, 5], [3, 4]]),
            # r free, x=y forced -> r=0
            (compute_domains_neq_imp, [(0, 1), (5, 5), (5, 5)], [], PROP_ENTAILMENT, [[0, 0], [5, 5], [5, 5]]),
            # r=1 steps x off c sitting on its lower bound
            (compute_domains_neq_c_imp, [(1, 1), (4, 9)], [4], PROP_ENTAILMENT, [[1, 1], [5, 9]]),
            # r=1, c strictly inside x -> no pruning (a hole cannot be represented)
            (compute_domains_neq_c_imp, [(1, 1), (0, 9)], [4], PROP_CONSISTENCY, [[1, 1], [0, 9]]),
            # r=1, x=c -> inconsistency
            (compute_domains_neq_c_imp, [(1, 1), (4, 4)], [4], PROP_INCONSISTENCY, None),
            # r free, x=c forced -> r=0
            (compute_domains_neq_c_imp, [(0, 1), (4, 4)], [4], PROP_ENTAILMENT, [[0, 0], [4, 4]]),
            # r free, c outside x -> entailed, r untouched (unlike full reif, r is NOT forced true)
            (compute_domains_neq_c_imp, [(0, 1), (5, 9)], [4], PROP_ENTAILMENT, [[0, 1], [5, 9]]),
            # r=1 snaps x onto the allowed values in range
            (compute_domains_member_imp, [(1, 1), (1, 9)], [0, 3, 6, 10], PROP_CONSISTENCY, [[1, 1], [3, 6]]),
            # r free, no allowed value in range -> r=0
            (compute_domains_member_imp, [(0, 1), (4, 5)], [0, 3, 6], PROP_ENTAILMENT, [[0, 0], [4, 5]]),
            # r free, x always in the set -> entailed, r untouched
            (compute_domains_member_imp, [(0, 1), (3, 4)], [0, 3, 4, 6], PROP_ENTAILMENT, [[0, 1], [3, 4]]),
            # r=0 -> vacuous, x keeps the allowed values
            (compute_domains_member_imp, [(0, 0), (0, 9)], [3], PROP_ENTAILMENT, [[0, 0], [0, 9]]),
        ],
    )
    def test_compute_domains(
        self,
        compute_fn: Callable,
        domains: list[Any],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[Any],
    ) -> None:
        self.assert_compute_domains(compute_fn, domains, parameters, consistency_result, expected_domains)
