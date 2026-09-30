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
import pytest

from nucs.constants import DOMAIN_MIN
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm, index
from nucs.heuristics.heuristics import VAR_HEURISTIC_DOM_WDEG
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm
from nucs.solvers.search import Search
from nucs.solvers.solver import OPTIM_PRUNE, OPTIM_RESET


class TestGolomb:
    @pytest.mark.parametrize(
        "mark_nb,i,j,idx", [(4, 0, 1, 0), (4, 0, 2, 1), (4, 0, 3, 2), (4, 1, 2, 3), (4, 1, 3, 4), (4, 2, 3, 5)]
    )
    def test_index(self, mark_nb: int, i: int, j: int, idx: int) -> None:
        assert index(mark_nb, i, j) == idx

    @pytest.mark.parametrize("mark_nb,length", [(4, 6), (5, 11), (6, 17), (7, 25), (8, 34), (9, 44), (10, 55)])
    def test_find_best(self, mark_nb: int, length: int) -> None:
        problem = GolombProblem(mark_nb)
        consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
        solver = BacktrackSolver(problem, consistency_algorithm=consistency_alg_golomb)
        solution = solver.find_best(problem.length_idx, DOMAIN_MIN)
        assert solution is not None
        assert solution[problem.length_idx] == length

    @pytest.mark.parametrize("mode", [OPTIM_PRUNE, OPTIM_RESET])
    @pytest.mark.parametrize("weight_decay", [1.0, 0.9])
    def test_find_best_dom_wdeg(self, mode: str, weight_decay: float) -> None:
        # the heuristic changes the tree, never the optimum
        problem = GolombProblem(7)
        solver = BacktrackSolver(
            problem, searches=[Search(var_heuristic=VAR_HEURISTIC_DOM_WDEG)], weight_decay=weight_decay
        )
        solution = solver.find_best(problem.length_idx, DOMAIN_MIN, mode=mode)
        assert solution is not None
        assert solution[problem.length_idx] == 25
