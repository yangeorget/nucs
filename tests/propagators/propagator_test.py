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

from nucs.constants import PROP_CONSISTENCY, PROP_FLAG_IDEMPOTENT, PROP_FLAG_REPORTS_CHANGES
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
