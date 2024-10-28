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
from typing import List, Tuple

from numpy.typing import NDArray


class ChoicePoints:
    def __init__(self) -> None:
        self.list: List[Tuple[NDArray, NDArray]] = []

    def size(self) -> int:
        return len(self.list)

    def pop(self) -> bool:
        if len(self.list) <= 1:
            return False
        self.list[0] = self.list.pop()
        return True

    def get_shr_domains(self) -> NDArray:
        return self.list[0][0]

    def get_not_entailed_propagators(self) -> NDArray:
        return self.list[0][1]

    def put(self, cp: Tuple[NDArray, NDArray]) -> None:
        self.list.append(cp)

    def clear(self) -> None:
        self.list.clear()
