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
import re

import pytest
from numba import int32, int64, njit, uint32  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_add, buckets_pop
from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.solvers.bc_algorithm import bc_algorithm
from nucs.solvers.consistency_algorithms import SIGN_CONSISTENCY_ALG


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
