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

from nucs.numba_helper import NUMBA_DISABLE_JIT

# A case that runs only with the JIT. Without the JIT, the CI runs the suite under coverage, and that step is most of
# the CI time: on 2026-11-13, the large example instances were 210 s of the 269 s of the suite on a fast machine, and
# the step took 1110 s on the 3.12 runner. Give this mark only to a large case of a test that keeps smaller cases
# without the JIT: they run the same code, so the coverage does not change, and the JIT run still checks the large
# case.
SLOW_WITHOUT_JIT = pytest.mark.skipif(
    bool(NUMBA_DISABLE_JIT), reason="too slow without the JIT; a smaller case of the same test runs the same code"
)
