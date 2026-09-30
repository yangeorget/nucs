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

import pytest

from nucs.solvers.restarts import (
    RESTART_CONSTANT,
    RESTART_GEOMETRIC,
    RESTART_LINEAR,
    RESTART_LUBY,
    RESTART_NONE,
    Restarts,
    luby,
    restart_limits,
)


class TestRestarts:
    def test_luby(self) -> None:
        assert [luby(i) for i in range(1, 16)] == [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]

    @pytest.mark.parametrize(
        "policy,scale,base,limits",
        [
            (RESTART_NONE, 1, 2.0, [-1, -1, -1, -1]),
            (RESTART_CONSTANT, 10, 2.0, [10, 10, 10, 10]),
            (RESTART_LINEAR, 10, 2.0, [10, 20, 30, 40]),
            (RESTART_GEOMETRIC, 10, 1.5, [10, 15, 22, 34]),
            (RESTART_LUBY, 10, 2.0, [10, 10, 20, 10]),
        ],
    )
    def test_restart_limits(self, policy: str, scale: int, base: float, limits: list[int]) -> None:
        assert list(itertools.islice(restart_limits(policy, scale, base), 4)) == limits

    @pytest.mark.parametrize(
        "policy,scale,base",
        [("fibonacci", 10, 2.0), (RESTART_LUBY, 0, 2.0), (RESTART_GEOMETRIC, 10, 1.0)],
    )
    def test_restart_limits_rejects_a_wrong_policy(self, policy: str, scale: int, base: float) -> None:
        with pytest.raises(ValueError):
            restart_limits(policy, scale, base)
        with pytest.raises(ValueError):  # at creation, before any search
            Restarts(policy, scale, base)

    def test_restarts_default_to_none(self) -> None:
        assert list(itertools.islice(Restarts().limits(), 3)) == [-1, -1, -1]

    def test_restarts_limits_start_again_at_each_call(self) -> None:
        """Each search starts the limits from the first one, so limits() gives a new iterator at each call."""
        restarts = Restarts(RESTART_LUBY, 10)
        assert list(itertools.islice(restarts.limits(), 3)) == [10, 10, 20]
        assert list(itertools.islice(restarts.limits(), 3)) == [10, 10, 20]
