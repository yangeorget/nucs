import numpy as np

from ncs.propagators.propagator import Propagator


class TestPropagator:
    def test_should_update_ok(self) -> None:
        propagator = Propagator([0, 1])
        propagator.triggers = np.array([[False, True], [True, False]])
        changes = np.array([[False, True], [True, False]])
        assert propagator.should_update(changes)

    def test_should_update_ko(self) -> None:
        propagator = Propagator([0, 1])
        propagator.triggers = np.array([[False, True], [True, False]])
        changes = np.array([[True, False], [False, True]])
        assert not propagator.should_update(changes)
