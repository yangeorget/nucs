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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from abc import abstractmethod
from multiprocessing import Queue
from typing import Optional

import enlighten
from numpy.typing import NDArray


class QueueSolver:
    """
    A solver which uses a queue.
    """

    @abstractmethod
    def get_progress_bar(self, manager: enlighten.Manager) -> Optional[enlighten.Counter]: ...

    @abstractmethod
    def get_statistics_as_array(self) -> NDArray: ...

    @abstractmethod
    def minimize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that minimizes a variable.
        :param variable_idx: the index of the variable to minimize
        :param processor_idx: the index of the processor running the minimizer
        :param solution_queue: the solution queue
        """
        ...

    @abstractmethod
    def maximize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that maximizes a variable.
        :param variable_idx: the index of the variable to maximizer
        :param processor_idx: the index of the processor running the maximizer
        :param solution_queue: the solution queue
        """
        ...

    @abstractmethod
    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solutions.
        :param processor_idx: the index of the processor
        :param solution_queue: the solution queue
        """
        ...
