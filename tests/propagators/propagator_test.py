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

from nucs.constants import PROP_CONSISTENCY, PROP_FLAG_IDEMPOTENT
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
        prop_state_arr = np.zeros(sum(_get_state_size(compute_domains_fct, domains_arr, parameters)), dtype=np.int32)
        status = compute_domains_fct(domains_arr, parameters_arr, prop_state_arr)
        # A propagator that is not idempotent is rescheduled by the engine after every call that changed a
        # domain, so its outcome is the outcome of that iteration rather than of any single call; asserting
        # one call would be asserting something the solver never observes.
        if not _is_idempotent(compute_domains_fct):
            while status == PROP_CONSISTENCY:
                previous = domains_arr.copy()
                status = compute_domains_fct(domains_arr, parameters_arr, prop_state_arr)
                if np.array_equal(previous, domains_arr):
                    break
        assert status == consistency_result
        if expected_domains:
            assert np.all(domains_arr == np.array(expected_domains))


def _is_idempotent(compute_domains_fct: Callable) -> bool:
    """
    Returns whether the propagator implemented by a compute_domains function reaches its own fixpoint in one
    call; unknown functions (a test-local one, say) are treated as idempotent.
    """
    for algorithm, fct in enumerate(COMPUTE_DOMAINS_FCTS):
        if fct is compute_domains_fct:
            return bool(ALGORITHM_FLAGS[algorithm] & PROP_FLAG_IDEMPOTENT)
    return True


def _get_state_size(compute_domains_fct: Callable, domains_arr: np.ndarray, parameters: list[int]) -> tuple[int, int]:
    """
    Returns the (trailed_nb, hint_nb) size of a compute_domains function's state block, by the same
    by-identity lookup as _is_idempotent; unknown functions (a test-local one, say) get none.
    """
    for algorithm, fct in enumerate(COMPUTE_DOMAINS_FCTS):
        if fct is compute_domains_fct:
            return GET_STATE_FCTS[algorithm](len(domains_arr), parameters)
    return 0, 0
