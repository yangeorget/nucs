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
# Copyright 2024 - Yan Georget
###############################################################################
from multiprocessing import Queue
from typing import List, Optional, Tuple

from numpy.typing import NDArray


class ChoicePoints:
    def is_empty(self) -> bool:
        return True

    def size(self) -> int:
        return 0

    def get(self) -> Optional[Tuple[NDArray, NDArray]]:
        return None

    def put(self, cp: Tuple[NDArray, NDArray]) -> None:
        pass

    def clear(self) -> None:
        pass


class ChoicePointList(ChoicePoints):
    def __init__(self) -> None:
        self.list: List[Tuple[NDArray, NDArray]] = []

    def is_empty(self) -> bool:
        return len(self.list) == 0

    def size(self) -> int:
        return len(self.list)

    def get(self) -> Optional[Tuple[NDArray, NDArray]]:
        return self.list.pop()

    def put(self, cp: Tuple[NDArray, NDArray]) -> None:
        self.list.append(cp)

    def clear(self) -> None:
        self.list.clear()


class ChoicePointQueue(ChoicePoints):
    def __init__(self, queue: Queue) -> None:
        self.queue = queue

    def is_empty(self) -> bool:
        return self.queue.empty()

    def size(self) -> int:
        return 1  # TODO fix

    def get(self) -> Optional[Tuple[NDArray, NDArray]]:
        return self.queue.get()

    def put(self, cp: Tuple[NDArray, NDArray]) -> None:
        self.queue.put(cp)

    def clear(self) -> None:
        while not self.queue.is_empty():
            self.queue.get()
