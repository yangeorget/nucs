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
from abc import abstractmethod
from typing import List, Tuple

from numpy.typing import NDArray


class ChoicePoints:
    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def get(self) -> Tuple[NDArray, NDArray]: ...

    @abstractmethod
    def put(self, cp: Tuple[NDArray, NDArray]) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class ChoicePointList(ChoicePoints):
    def __init__(self) -> None:
        self.list: List[Tuple[NDArray, NDArray]] = []

    def is_empty(self) -> bool:
        return len(self.list) == 0

    def size(self) -> int:
        return len(self.list)

    def get(self) -> Tuple[NDArray, NDArray]:
        return self.list.pop()

    def put(self, cp: Tuple[NDArray, NDArray]) -> None:
        self.list.append(cp)

    def clear(self) -> None:
        self.list.clear()
