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
from typing import Any, Dict, List

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ALLDIFFERENT


def starts_word(letter: str, words: List[str]) -> bool:
    for word in words:
        if word.startswith(letter):
            return True
    return False


class CryptarithmeticProblem(Problem):
    """
    A general cryptarithmetic problem where letters represent unique values.

    Supports addition constraints, multi-digit column arithmetic where letter values are digits:
    e.g. DONALD + GERALD = ROBERT
    """

    def __init__(self, dataset: Dict) -> None:
        self.letters = dataset["letters"]
        n = len(self.letters)
        letter_to_idx = {letter: idx for idx, letter in enumerate(self.letters)}
        all_words = []
        for words, result_word in dataset.get("additions", []):
            all_words += words
            all_words.append(result_word)
        super().__init__([(1, 9) if starts_word(letter, all_words) else (0, 9) for letter in self.letters])
        for words, result_word in dataset.get("additions", []):
            coefficients: Dict[str, int] = {}
            for word in words:
                coefficient = 10 ** (len(word) - 1)
                for character in word:
                    coefficients[character] = coefficients.get(character, 0) + coefficient
                    coefficient //= 10
            coefficient = 10 ** (len(result_word) - 1)
            for character in result_word:
                coefficients[character] = coefficients.get(character, 0) - coefficient
                coefficient //= 10
            addition_letters = sorted(coefficients.keys(), key=lambda letter: letter_to_idx[letter])
            self.add_propagator(
                ALG_AFFINE_EQ,
                [letter_to_idx[letter] for letter in addition_letters],
                [coefficients[letter] for letter in addition_letters] + [0],
            )
        self.add_propagator(ALG_ALLDIFFERENT, list(range(n)))

    def solution_as_printable(self, solution: NDArray) -> Any:
        return {letter: int(solution[i]) for i, letter in enumerate(self.letters)}
