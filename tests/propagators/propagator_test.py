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
from collections.abc import Callable

import numpy as np

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_FLAG_IDEMPOTENT,
    PROP_FLAG_REPORTS_CHANGES,
    PROP_INCONSISTENCY,
)
from nucs.propagators.propagators import ALGORITHM_FLAGS, COMPUTE_DOMAINS_FCTS, GET_STATE_FCTS


class PropagatorTest:
    def assert_compute_domains(
        self,
        compute_domains_fct: Callable,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        domains_arr = np.array(
            [(domain, domain) if isinstance(domain, int) else domain for domain in domains], dtype=np.int32
        )
        parameters_arr = np.array(parameters, dtype=np.int32)
        # a fresh, zeroed state block, correctly sized by the propagator's own get_state_fct -- (0, 0), and
        # so an empty array, for every propagator that doesn't declare one
        trailed_nb, hint_nb = _get_state_size(compute_domains_fct, domains_arr, parameters)
        prop_state_arr = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
        report_idx = trailed_nb if _reports_changes(compute_domains_fct) else -1
        status = self._call(compute_domains_fct, domains_arr, parameters_arr, prop_state_arr, report_idx)
        # A propagator that is not idempotent is rescheduled by the engine after every call that changed a
        # domain, so its outcome is the outcome of that iteration rather than of any single call; asserting
        # one call would be asserting something the solver never observes.
        if not _is_idempotent(compute_domains_fct):
            while status == PROP_CONSISTENCY:
                previous = domains_arr.copy()
                status = self._call(compute_domains_fct, domains_arr, parameters_arr, prop_state_arr, report_idx)
                if np.array_equal(previous, domains_arr):
                    break
        assert status == consistency_result
        if expected_domains:
            assert np.all(domains_arr == np.array(expected_domains))

    @staticmethod
    def _call(
        compute_domains_fct: Callable,
        domains_arr: np.ndarray,
        parameters_arr: np.ndarray,
        prop_state_arr: np.ndarray,
        report_idx: int,
    ) -> int:
        """
        Calls a propagator the way bc_algorithm does, and holds it to what it reports about itself.

        A propagator that declares PROP_FLAG_REPORTS_CHANGES answers, in the first cell of its state
        block's hint suffix, whether it wrote any domain; the engine takes a 0 there as licence to skip
        update_domains entirely. Nothing in the engine checks that answer -- it cannot, since checking is
        the scan it is trying to avoid -- so a propagator that reports 0 after narrowing something loses
        that pruning silently, and the search quietly explores more of the tree, or returns a wrong answer
        if the write it dropped was the one that failed the node. This is where that contract is checked,
        against the propagator's own curated cases, on every call rather than only the first.

        Only the unsafe direction is an error. Reporting a change that did not happen costs the scan the
        propagator would have paid anyway, which is why the engine pre-sets the cell to 1: a propagator
        that forgets to answer is merely slow.
        """
        if report_idx >= 0:
            prop_state_arr[report_idx] = 1  # what the engine writes before every call
        before = domains_arr.copy()
        status = compute_domains_fct(domains_arr, parameters_arr, prop_state_arr)
        if report_idx >= 0 and status == PROP_CONSISTENCY and prop_state_arr[report_idx] == 0:
            assert np.array_equal(before, domains_arr), (
                f"{compute_domains_fct.__name__} reported no change but narrowed a domain: "
                f"{before.tolist()} -> {domains_arr.tolist()}"
            )
        return status

    def assert_live_set_is_sound(
        self,
        compute_domains_fct: Callable,
        domains: np.ndarray,
        parameters: list[int],
        rng: np.random.Generator,
        backtrack: bool,
    ) -> None:
        """
        Holds a propagator that carries a live set across calls to what a propagator without one concludes.

        A propagator whose state block summarises its variables -- which of them are still undecided, and
        what the decided ones have already contributed -- rebuilds none of that per call, which is sound
        only because domains narrow monotonically inside a branch. So the thing to check is not any one
        call but a *chain* of them: narrow a little at a time, let one state block follow the chain, and
        demand at each step that it agrees -- status, every bound, and the change report -- with a
        propagator handed the same domains and a block it has never seen.

        With backtrack set, the chain also does what the solver does at a choice point: descend, then
        restore the trailed prefix and the bounds and descend differently. Only the prefix is restored,
        because only the prefix is trailed -- the permutation behind it is left exactly as the abandoned
        subtree permuted it, which is sound only because a departing element is parked past the end of the
        live prefix rather than dropped. Nothing else in these tests can see that.

        :param compute_domains_fct: the compute_domains function under test
        :type compute_domains_fct: Callable
        :param domains: the starting domains, narrowed in place
        :type domains: np.ndarray
        :param parameters: the propagator parameters
        :type parameters: list[int]
        :param rng: the source of the narrowings
        :type rng: np.random.Generator
        :param backtrack: whether the chain restores a choice point partway through
        :type backtrack: bool
        """
        trailed_nb, hint_nb = _get_state_size(compute_domains_fct, domains, parameters)
        report_idx = trailed_nb if _reports_changes(compute_domains_fct) else -1
        parameters_arr = np.array(parameters, dtype=np.int32)
        warm_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
        n = len(domains)

        def narrow(doms: np.ndarray) -> None:
            # only ever narrow: a widening is something the engine never does inside a branch, and every
            # live set here is unsound without that
            i = int(rng.integers(0, n))
            if doms[i, DOMAIN_MIN] < doms[i, DOMAIN_MAX]:
                if rng.integers(0, 2):
                    doms[i, DOMAIN_MIN] += 1
                else:
                    doms[i, DOMAIN_MAX] -= 1

        def step(doms: np.ndarray) -> int:
            cold_domains = doms.copy()
            cold_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
            if report_idx >= 0:
                warm_state[report_idx] = cold_state[report_idx] = 1  # what the engine pre-sets
            warm_status = compute_domains_fct(doms, parameters_arr, warm_state)
            cold_status = compute_domains_fct(cold_domains, parameters_arr, cold_state)
            assert warm_status == cold_status
            assert np.array_equal(doms, cold_domains)
            if report_idx >= 0:
                assert warm_state[report_idx] == cold_state[report_idx]
            return warm_status

        node_domains = domains.copy()
        for _ in range(3):
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                return
        if not backtrack:
            for _ in range(9):
                narrow(node_domains)
                if step(node_domains) != PROP_CONSISTENCY:
                    return
            return
        trailed_mark = warm_state[:trailed_nb].copy()  # the choice point: the trail holds this and the bounds
        domains_mark = node_domains.copy()
        for _ in range(6):  # descend, permuting the live array as it goes
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                break
        warm_state[:trailed_nb] = trailed_mark  # backtrack: exactly what the trail undoes, and no more
        node_domains = domains_mark.copy()
        for _ in range(6):  # and away down a different branch
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                return

    def assert_sound_against_brute_force(
        self,
        compute_domains_fct: Callable,
        bounds: list[tuple[int, int]],
        parameters: list[int],
        is_solution: Callable[[tuple[int, ...]], bool],
        assumption: Callable[[tuple[int, ...]], bool] | None = None,
    ) -> None:
        """
        Holds one call of a propagator, from a fresh state block, to what enumerating the box says.

        The propagator must never prune a value of some solution, never fail while a solution exists, never leave
        a domain empty without failing, and only claim entailment when every point left in the narrowed box is a solution. When it declares itself
        idempotent, a second call with the same state block must narrow nothing, since the engine never wakes
        it on its own changes; otherwise it is run to its fixpoint, as the engine would. The change report is
        checked on every call.

        A propagator that relies on another constraint being posted alongside (element_l_eq_alldifferent on an
        alldifferent, say) is held to the conjunction: only the points satisfying the assumption count, and a box
        without any is skipped, since the other constraint fails it.

        :param compute_domains_fct: the compute_domains function under test
        :type compute_domains_fct: Callable
        :param bounds: the domains, small enough to enumerate
        :type bounds: list[tuple[int, int]]
        :param parameters: the propagator parameters
        :type parameters: list[int]
        :param is_solution: whether a point of the box satisfies the constraint
        :type is_solution: Callable[[tuple[int, ...]], bool]
        :param assumption: whether a point satisfies the constraints the propagator relies on, None for none
        :type assumption: Callable[[tuple[int, ...]], bool] | None
        """
        points = [
            point
            for point in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds])
            if assumption is None or assumption(point)
        ]
        if not points:
            return
        solutions = [point for point in points if is_solution(point)]
        domains = np.array(bounds, dtype=np.int32)
        parameters_arr = np.array(parameters, dtype=np.int32)
        trailed_nb, hint_nb = _get_state_size(compute_domains_fct, domains, parameters)
        state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
        report_idx = trailed_nb if _reports_changes(compute_domains_fct) else -1
        status = self._call(compute_domains_fct, domains, parameters_arr, state, report_idx)
        idempotent = _is_idempotent(compute_domains_fct)
        if not idempotent:
            while status == PROP_CONSISTENCY:
                previous = domains.copy()
                status = self._call(compute_domains_fct, domains, parameters_arr, state, report_idx)
                if np.array_equal(previous, domains):
                    break
        if status == PROP_INCONSISTENCY:
            assert not solutions, f"declared inconsistent but {solutions[0]} is a solution: {bounds} {parameters}"
            return
        assert np.all(domains[:, DOMAIN_MIN] <= domains[:, DOMAIN_MAX]), (
            f"left a domain empty without failing: {bounds} {parameters} -> {domains.tolist()}"
        )
        for solution in solutions:
            for i, value in enumerate(solution):
                assert domains[i, DOMAIN_MIN] <= value <= domains[i, DOMAIN_MAX], (
                    f"pruned x[{i}]={value} of {solution}: {bounds} {parameters} -> {domains.tolist()}"
                )
        if status == PROP_ENTAILMENT:
            for point in itertools.product(*[range(lo, hi + 1) for lo, hi in domains.tolist()]):
                assert (assumption is not None and not assumption(point)) or is_solution(point), (
                    f"declared entailed but {point} is not a solution: {bounds} {parameters}"
                )
        elif idempotent:
            fixpoint = domains.copy()
            status = self._call(compute_domains_fct, domains, parameters_arr, state, report_idx)
            assert status != PROP_INCONSISTENCY and np.array_equal(domains, fixpoint), (
                f"not idempotent: {bounds} {parameters} -> {fixpoint.tolist()} -> {domains.tolist()}"
            )


def random_bounds(rng: random.Random, k: int, lo: int, hi: int) -> list[tuple[int, int]]:
    """
    Returns k random intervals within [lo, hi], for brute-force tests.

    :param rng: the source of the intervals
    :type rng: random.Random
    :param k: the number of intervals
    :type k: int
    :param lo: the smallest value
    :type lo: int
    :param hi: the largest value
    :type hi: int

    :return: the intervals
    :rtype: list[tuple[int, int]]
    """
    bounds = []
    for _ in range(k):
        a = rng.randint(lo, hi)
        bounds.append((a, rng.randint(a, hi)))
    return bounds


def _is_idempotent(compute_domains_fct: Callable) -> bool:
    """
    Returns whether the propagator implemented by a compute_domains function reaches its own fixpoint in one
    call; unknown functions (a test-local one, say) are treated as idempotent.
    """
    for algorithm, fct in enumerate(COMPUTE_DOMAINS_FCTS):
        if fct is compute_domains_fct:
            return bool(ALGORITHM_FLAGS[algorithm] & PROP_FLAG_IDEMPOTENT)
    return True


def _reports_changes(compute_domains_fct: Callable) -> bool:
    """
    Returns whether the propagator answers the engine about what it changed, by the same by-identity
    lookup as _is_idempotent; unknown functions (a test-local one, say) do not.
    """
    for algorithm, fct in enumerate(COMPUTE_DOMAINS_FCTS):
        if fct is compute_domains_fct:
            return bool(ALGORITHM_FLAGS[algorithm] & PROP_FLAG_REPORTS_CHANGES)
    return False


def _get_state_size(compute_domains_fct: Callable, domains_arr: np.ndarray, parameters: list[int]) -> tuple[int, int]:
    """
    Returns the (trailed_nb, hint_nb) size of a compute_domains function's state block, by the same
    by-identity lookup as _is_idempotent; unknown functions (a test-local one, say) get none.
    """
    for algorithm, fct in enumerate(COMPUTE_DOMAINS_FCTS):
        if fct is compute_domains_fct:
            return GET_STATE_FCTS[algorithm](len(domains_arr), parameters)
    return 0, 0
