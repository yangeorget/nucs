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
import os
import subprocess
import sys

import pytest


class TestNumbaHelper:
    @pytest.mark.parametrize("value,disabled", [("0", False), ("1", True)])
    def test_numba_disable_jit_follows_numba(self, value: str, disabled: bool) -> None:
        # the flag is read at import, hence a fresh interpreter; "0" once read as the true string "0", which ran the
        # no-JIT paths under the JIT
        result = subprocess.run(
            [sys.executable, "-c", "from nucs.numba_helper import NUMBA_DISABLE_JIT; print(NUMBA_DISABLE_JIT)"],
            capture_output=True,
            text=True,
            env={**os.environ, "NUMBA_DISABLE_JIT": value},
            check=True,
        )
        assert result.stdout.strip() == str(disabled)
