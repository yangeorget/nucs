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
from typing import List, Any

from numpy.typing import NDArray

from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_CUMULATIVE, ALG_DIFFN
from nucs.solvers.backtrack_solver import Search


class SquarePlacementProblem(Problem):
    """
    The perfect square placement problem (CSPLIB problem #9).

    A set of squares of given integer sizes must be packed, edges parallel to the border and without overlap,
    into a ``width`` by ``height`` master rectangle (a square when ``width == height``). The model has one
    bottom-left corner ``(x_i, y_i)`` per square; the squares are kept disjoint by a global diffn constraint,
    and two redundant cumulative constraints make the projection on each axis tight: scanning along x, the
    squares covering a column must fit within the height, and symmetrically along y within the width. These
    redundant resource constraints are what make the model propagate strongly.

    CSPLIB problem #9 - https://www.csplib.org/Problems/prob009/
    """

    def __init__(self, width: int, height: int, sizes: List[int]) -> None:
        """
        Initializes the problem.

        :param width: the width of the master rectangle
        :type width: int
        :param height: the height of the master rectangle
        :type height: int
        :param sizes: the sizes of the squares to place
        :type sizes: List[int]
        """
        self.width = width
        self.height = height
        self.sizes = sorted(sizes, reverse=True)  # largest first: stronger propagation and a cleaner symmetry break
        n = len(self.sizes)
        self.square_nb = n
        # x_i in [0, width - s_i], y_i in [0, height - s_i]
        x_domains = [(0, width - s) for s in self.sizes]
        y_domains = [(0, height - s) for s in self.sizes]
        # symmetry breaking: pin the largest square's bottom-left corner to the lower-left quadrant
        x_domains[0] = (0, (width - self.sizes[0]) // 2)
        y_domains[0] = (0, (height - self.sizes[0]) // 2)
        super().__init__(x_domains + y_domains)
        xs = list(range(n))
        ys = list(range(n, 2 * n))
        # diffn: the squares (rectangles of size s_i x s_i) do not overlap
        self.add_propagator(ALG_DIFFN, xs + ys, self.sizes + self.sizes)
        # cumulative along x: at every column the heights of the covering squares fit within the height
        self.add_propagator(ALG_CUMULATIVE, xs, self.sizes + self.sizes + [height])
        # cumulative along y: at every row the widths of the covering squares fit within the width
        self.add_propagator(ALG_CUMULATIVE, ys, self.sizes + self.sizes + [width])

    def recommended_searches(self) -> List[Search]:
        """
        Returns the recommended search for this problem.

        It fills the master left to right then bottom to top: a first search places the x coordinates, a second
        the y coordinates, each time choosing the square that can go leftmost (resp. lowest) and committing it
        to that position (smallest-minimal-value variable heuristic with the min_value domain heuristic). The
        cumulative constraints make this packing order propagate strongly. Pass it to a solver with
        ``BacktrackSolver(problem, searches=problem.recommended_searches())``.

        :return: a two-search sequential search, x coordinates then y coordinates
        :rtype: List[Search]
        """
        n = self.square_nb
        return [
            Search(range(n), VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE, dom_heuristic=DOM_HEURISTIC_MIN_VALUE),
            Search(range(n, 2 * n), VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE, dom_heuristic=DOM_HEURISTIC_MIN_VALUE),
        ]

    def x(self, square: int) -> int:
        """
        Returns the variable index of the x coordinate of a square's bottom-left corner.

        :param square: the square index
        :type square: int

        :return: the x-coordinate variable index
        :rtype: int
        """
        return square

    def y(self, square: int) -> int:
        """
        Returns the variable index of the y coordinate of a square's bottom-left corner.

        :param square: the square index
        :type square: int

        :return: the y-coordinate variable index
        :rtype: int
        """
        return self.square_nb + square

    def solution_as_printable(self, solution: NDArray) -> Any:
        sol_as_printable = ""
        for i in range(self.square_nb):
            sol_as_printable += f"square {self.sizes[i]}: ({int(solution[self.x(i)])}, {int(solution[self.y(i)])})\n"
        return sol_as_printable
