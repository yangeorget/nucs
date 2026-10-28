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
The interruption of a search: a two-cell array that the compiled search reads at every node, one cell for each of
its two writers, interrupt() and a deadline.
"""

import threading
from collections.abc import Callable

import numpy as np

# the cells of the array, one for each writer, which writes 1 there; the compiled search stops when either is 1
INTERRUPTION_EXTERNAL = 0  # interrupt(): final, every later search stops too
INTERRUPTION_DEADLINE = 1  # a timeout: cleared once the search it belongs to is over
INTERRUPTION_WIDTH = 2  # the number of cells


class Interrupter:
    """
    Asks a search that runs in compiled code to stop at its next node.

    A call of run_search_loop returns to Python only at a solution, so cells that it reads itself at every node are
    the only way to stop it in between. interrupt() writes its cell for good, and a deadline writes its own cell for
    one search only. Since each writer has its own cell, the clearing of a deadline cannot erase an interrupt().
    """

    def __init__(self) -> None:
        """
        Initializes an interrupter that asks nothing yet.
        """
        # read by run_search_loop at every node, and written possibly from another thread
        self.cells = np.zeros(INTERRUPTION_WIDTH, dtype=np.int32)
        # serializes a deadline timer and the clearing of that deadline, so that the timer cannot write after
        # the clearing; interrupt() takes no lock, since it may run as a signal handler on a thread that holds it
        self.deadline_lock = threading.Lock()

    def interrupt(self) -> None:
        """
        Asks the search to stop at its next node, and every later search to stop at once.

        Safe to call from another thread or from a signal handler: it takes no lock, it only writes its cell.
        """
        self.cells[INTERRUPTION_EXTERNAL] = 1

    def arm_deadline(self, timeout: float | None) -> Callable[[], None]:
        """
        Starts a timer that stops the search at its next node once the timeout has elapsed.

        A check of the clock between solutions would not be enough: a call of run_search_loop that finds no
        solution -- a proof of optimality, an infeasible subtree -- never returns to Python, and would run past
        the budget for as long as it lasts. So the timer writes a cell that the compiled search reads, and it is
        the only check of the budget: a consumer that keeps a solution past it stops the next call at its first
        node. Unlike interrupt(), the deadline belongs to one search: the returned function cancels the timer and
        clears its cell, which leaves the cell of interrupt() as it is.

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
                if armed[0]:
                    self.cells[INTERRUPTION_DEADLINE] = 1

        timer = threading.Timer(max(timeout, 0.0), expire)
        timer.daemon = True
        timer.start()

        def disarm() -> None:
            timer.cancel()
            with self.deadline_lock:
                armed[0] = False
                self.cells[INTERRUPTION_DEADLINE] = 0

        return disarm
