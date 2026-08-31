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

from nucs.constants import PROP_CONSISTENCY
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, IDEMPOTENCIES


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
        status = compute_domains_fct(domains_arr, parameters_arr)
        # A propagator that is not idempotent is rescheduled by the engine after every call that changed a
        # domain, so its outcome is the outcome of that iteration rather than of any single call; asserting
        # one call would be asserting something the solver never observes.
        if not _is_idempotent(compute_domains_fct):
            while status == PROP_CONSISTENCY:
                previous = domains_arr.copy()
                status = compute_domains_fct(domains_arr, parameters_arr)
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
            return IDEMPOTENCIES[algorithm]
    return True
