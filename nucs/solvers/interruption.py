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
"""
The interruption of a search: a one-cell array that the compiled search reads at every node, and its two writers,
interrupt() and a deadline.
"""

import threading
from collections.abc import Callable

import numpy as np

# what the cell holds; the compiled search stops on anything non-zero
INTERRUPTION_NONE = 0
INTERRUPTION_EXTERNAL = 1  # interrupt(): final, every later search stops too
INTERRUPTION_DEADLINE = 2  # a timeout: cleared once the search it belongs to is over


class Interruption:
    """
    Asks a search that runs in compiled code to stop at its next node.

    A call of solve_one_step returns to Python only at a solution, so a cell that it reads itself at every node is
    the only way to stop it in between. interrupt() writes the cell for good, and a deadline writes it for one
    search only.
    """

    def __init__(self) -> None:
        """
        Initializes an interruption that asks nothing yet.
        """
        # read by solve_one_step at every node, and written possibly from another thread
        self.cell = np.zeros((1,), dtype=np.int32)
        # set by interrupt() before it writes the cell, and never cleared: the clearing of a deadline restores
        # INTERRUPTION_EXTERNAL from it, since interrupt() may run between any two bytecodes of that clearing
        self.interrupted = False
        # serializes a deadline timer and the clearing of that deadline, so that the timer cannot write after
        # the clearing; interrupt() takes no lock, since it may run as a signal handler on a thread that holds it
        self.deadline_lock = threading.Lock()

    def interrupt(self) -> None:
        """
        Asks the search to stop at its next node, and every later search to stop at once.

        Safe to call from another thread or from a signal handler: it takes no lock, it only sets a flag and writes
        the cell.
        """
        self.interrupted = True  # first: the clearing of a deadline restores the cell from it
        self.cell[0] = INTERRUPTION_EXTERNAL

    def arm_deadline(self, timeout: float | None) -> Callable[[], None]:
        """
        Starts a timer that stops the search at its next node once the timeout has elapsed.

        A check of the clock between solutions would not be enough: a call of solve_one_step that finds no
        solution -- a proof of optimality, an infeasible subtree -- never returns to Python, and would run past
        the budget for as long as it lasts. So the timer writes the same cell as interrupt(), and it is the only
        check of the budget: a consumer that keeps a solution past it stops the next call at its first node.
        Unlike interrupt(), the deadline belongs to one search: the returned function cancels the timer and clears
        what it wrote, leaving an external interruption in place.

        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: the function to call once the search is over
        :rtype: Callable[[], None]
        """
        if timeout is None:
            return lambda: None
        armed = [True]  # a callback already running when the timer is cancelled must not write after disarm

        def expire() -> None:
            with self.deadline_lock:
                if armed[0] and self.cell[0] == INTERRUPTION_NONE:
                    self.cell[0] = INTERRUPTION_DEADLINE

        timer = threading.Timer(max(timeout, 0.0), expire)
        timer.daemon = True
        timer.start()

        def disarm() -> None:
            timer.cancel()
            with self.deadline_lock:
                armed[0] = False
                # a test of the cell before the clear would erase an interrupt() that runs between the two:
                # clear it, then restore the external interruption from the flag that interrupt() sets first
                self.cell[0] = INTERRUPTION_NONE
                if self.interrupted:
                    self.cell[0] = INTERRUPTION_EXTERNAL

        return disarm
