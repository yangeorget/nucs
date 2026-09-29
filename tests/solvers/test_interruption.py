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
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import CodeType, FrameType

from nucs.solvers.interruption import INTERRUPTION_DEADLINE, INTERRUPTION_EXTERNAL, INTERRUPTION_NONE, Interruption


@contextmanager
def call_before_bytecode(code: CodeType, position: int, callback: Callable[[], None]) -> Iterator[list[int]]:
    """
    Traces the frames of a code object and calls a function before one of its bytecodes, as a signal handler can run.

    :param code: the code object whose frames are traced
    :type code: CodeType
    :param position: the index of the bytecode before which the function is called, -1 for no call
    :type position: int
    :param callback: the function to call
    :type callback: Callable[[], None]

    :return: a one-cell list that holds the number of bytecodes the traced frames ran, once the block is over
    :rtype: Iterator[list[int]]
    """
    opcode_nb = [0]

    def trace_opcodes(frame: FrameType, event: str, _arg: object) -> Callable | None:
        frame.f_trace_opcodes = True  # ignored when set at the call event, from Python 3.14
        if event == "opcode":
            if opcode_nb[0] == position:
                callback()
            opcode_nb[0] += 1
        return trace_opcodes

    def trace_calls(frame: FrameType, _event: str, _arg: object) -> Callable | None:
        return trace_opcodes if frame.f_code is code else None

    # Python 3.12 turns on opcode events only at a settrace that follows the first f_trace_opcodes of the process:
    # without this, the first traced frame gets none
    frame = sys._getframe()
    frame.f_trace_opcodes = True
    frame.f_trace_opcodes = False
    previous_trace = sys.gettrace()
    sys.settrace(trace_calls)
    try:
        yield opcode_nb
    finally:
        sys.settrace(previous_trace)


class TestInterruption:
    def test_disarm_before_the_timeout_cancels_the_deadline(self) -> None:
        """A deadline disarmed before it expires never writes the cell, so it cannot stop a later search."""
        interruption = Interruption()
        disarm = interruption.arm_deadline(0.05)
        disarm()
        time.sleep(0.1)  # past the timeout: a timer still armed would have written the cell by now
        assert interruption.cell[0] == INTERRUPTION_NONE

    def test_interrupt_inside_disarm_is_kept(self) -> None:
        """An interrupt() that runs in the middle of disarm, as a signal handler can, is not lost."""
        # a signal handler runs on the main thread between two bytecodes, even while that thread holds a lock:
        # call interrupt() at each bytecode of disarm in turn, the way such a handler can
        opcode_nb = self.interrupt_inside_disarm(-1)
        assert opcode_nb > 0
        for position in range(opcode_nb):
            self.interrupt_inside_disarm(position)

    def interrupt_inside_disarm(self, position: int) -> int:
        """
        Calls interrupt() at one bytecode of the disarm of an expired deadline and checks the interruption stays.

        :param position: the index of the bytecode of disarm before which interrupt() runs, -1 for no interrupt()
        :type position: int

        :return: the number of bytecodes that disarm ran
        :rtype: int
        """
        interruption = Interruption()
        disarm = interruption.arm_deadline(0.0)
        start = time.monotonic()
        while interruption.cell[0] != INTERRUPTION_DEADLINE and time.monotonic() - start < 1:
            time.sleep(0.001)
        assert interruption.cell[0] == INTERRUPTION_DEADLINE
        with call_before_bytecode(disarm.__code__, position, interruption.interrupt) as opcode_nb:
            disarm()
        expected = INTERRUPTION_NONE if position < 0 else INTERRUPTION_EXTERNAL
        assert interruption.cell[0] == expected, f"interrupt() at bytecode {position} of disarm"
        return opcode_nb[0]
