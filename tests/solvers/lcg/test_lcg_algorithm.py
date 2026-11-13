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
import pytest
from numba import int32, int64  # type: ignore

from nucs.numba_helper import NUMBA_DISABLE_JIT
from nucs.solvers.backtrack.consistency_algorithms import SIGN_CONSISTENCY_ALG
from nucs.solvers.lcg.lcg_algorithm import lcg_algorithm
from tests.solvers.backtrack.test_bc_algorithm import assert_compiled_without_refcounts

# the signature of the consistency algorithms, followed by the implication trail (itrail, itrail_top and itrail_last)
# and failed_propagator
SIGN_LCG_ALGORITHM = int64(*SIGN_CONSISTENCY_ALG.args, int32[:, ::1], int32[::1], int32[::1], int32[::1])


@pytest.mark.skipif(bool(NUMBA_DISABLE_JIT), reason="inspects the compiled code, which only exists under the JIT")
class TestLcgAlgorithmCompilation:
    def test_is_compiled_without_the_reference_counting_runtime(self) -> None:
        """
        Pins the private _nrt option on lcg_algorithm, as on bc_algorithm.
        """
        assert lcg_algorithm.targetoptions["_nrt"] is False

    def test_compiled_body_has_no_refcount_calls(self) -> None:
        """
        Checks that the compiled loop holds no refcount call, whatever compiled the functions it calls.
        """
        assert_compiled_without_refcounts(lcg_algorithm, "lcg_algorithm", SIGN_LCG_ALGORITHM)
