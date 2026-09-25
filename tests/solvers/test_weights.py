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

from nucs.solvers.weights import WEIGHTS_RESCALE_LIMIT, weights_bump, weights_init


class TestWeights:
    def test_weights_init(self) -> None:
        # each propagator starts at 1, then come the increment and its growth
        assert weights_init(3).tolist() == [1, 1, 1, 1, 1]
        assert weights_init(2, 0.5).tolist() == [1, 1, 1, 2]

    @pytest.mark.parametrize("decay", [0.0, -0.5, 1.5])
    def test_weights_init_rejects_a_decay_out_of_range(self, decay: float) -> None:
        with pytest.raises(ValueError):
            weights_init(2, decay)

    def test_weights_bump_without_decay_counts_the_failures(self) -> None:
        weights = weights_init(3)
        for propagator in [0, 2, 2]:
            weights_bump(weights, propagator)
        assert weights.tolist() == [2, 1, 3, 1, 1]

    def test_weights_bump_with_decay_prefers_the_recent_failures(self) -> None:
        # with a decay of 0.5, a failure counts twice as much as the one before it
        weights = weights_init(2, 0.5)
        weights_bump(weights, 0)
        weights_bump(weights, 1)
        assert weights.tolist() == [2, 3, 4, 2]

    def test_weights_bump_rescales_and_keeps_the_ratios(self) -> None:
        weights = weights_init(2, 0.5)
        weights[2] = WEIGHTS_RESCALE_LIMIT / 3  # the next increment passes the limit
        weights_bump(weights, 0)
        assert weights[2] < WEIGHTS_RESCALE_LIMIT
        # before the rescale: weight 0 = 1 + limit/3, weight 1 = 1, increment = 2 * limit/3
        assert weights[0] / weights[2] == pytest.approx(
            (1 + WEIGHTS_RESCALE_LIMIT / 3) / (2 * WEIGHTS_RESCALE_LIMIT / 3)
        )
        assert weights[1] / weights[2] == pytest.approx(1 / (2 * WEIGHTS_RESCALE_LIMIT / 3))
