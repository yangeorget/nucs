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
from typing import Any, Dict

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT


class AlphanumericProblem(Problem):
    """
    A general alphanumeric problem where letters represent unique values.

    The sum of each letter's value in a word equals a target:
    e.g. ALPHA puzzle: BALLET=45, CELLO=43, ...
    """

    def __init__(self, dataset: Dict) -> None:
        self.letters = dataset["letters"]
        letter_to_idx = {letter: idx for idx, letter in enumerate(self.letters)}
        n = len(self.letters)
        super().__init__([dataset["domain"]] * n)
        for word, word_sum in dataset.get("word_sums", []):
            word_counts: Dict[str, int] = {}
            for character in word:
                word_counts[character] = word_counts.get(character, 0) + 1
            word_letters = sorted(word_counts.keys(), key=lambda letter: letter_to_idx[letter])
            self.add_propagator(
                ALG_AFFINE_EQ,
                [letter_to_idx[letter] for letter in word_letters],
                [word_counts[letter] for letter in word_letters] + [word_sum],
            )
        self.add_propagator(ALG_ALLDIFFERENT, list(range(n)))

    def solution_as_printable(self, solution: NDArray) -> Any:
        return {letter: int(solution[i]) for i, letter in enumerate(self.letters)}
