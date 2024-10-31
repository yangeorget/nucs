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
import pytest

from nucs.constants import MIN
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm, index, init_domains
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm


class TestGolomb:

    @pytest.mark.parametrize(
        "mark_nb,i,j,idx", [(4, 0, 1, 0), (4, 0, 2, 1), (4, 0, 3, 2), (4, 1, 2, 3), (4, 1, 3, 4), (4, 2, 3, 5)]
    )
    def test_index(self, mark_nb: int, i: int, j: int, idx: int) -> None:
        assert index(mark_nb, i, j) == idx

    def test_init_domains(self) -> None:
        domains = init_domains(6, 4)
        assert domains[:, MIN].tolist() == [1, 3, 6, 1, 3, 1]

    @pytest.mark.parametrize("mark_nb,length", [(4, 6), (5, 11), (6, 17), (7, 25), (8, 34), (9, 44)])
    def test_golomb(self, mark_nb: int, length: int) -> None:
        problem = GolombProblem(mark_nb)
        consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
        solver = BacktrackSolver(problem, consistency_alg_idx=consistency_alg_golomb)
        solution = solver.minimize(problem.length_idx)
        assert solution is not None
        assert solution[problem.length_idx] == length
