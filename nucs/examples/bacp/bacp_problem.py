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
from nucs.propagators.propagators import (
    ALG_LINEAR_EQ_C,
    ALG_EQ_C_REIF,
    ALG_MAX_EQ,
    ALG_GCC,
    ALG_LEQ_C,
)


class BACPProblem(Problem):
    """
    The Balanced Academic Curriculum Problem.

    CSPLIB problem #30 - https://www.csplib.org/Problems/prob030/

    n_courses are assigned to n_periods. Each course has a load.
    For each period, the number of courses and the total load are bounded.
    A prerequisite [a, b] means course b is a prerequisite of course a, so b is taken in an earlier period than a.
    The objective is to minimize the maximum load over all periods.
    """

    def __init__(self, dataset: Dict) -> None:
        """
        Inits the problem.

        :param dataset: the dataset
        :type dataset: dict
        """
        self.n_courses: int = dataset["n_courses"]
        self.n_periods: int = dataset["n_periods"]
        load_lb: int = dataset["load_per_period_lb"]
        load_ub: int = dataset["load_per_period_ub"]
        courses_lb: int = dataset["courses_per_period_lb"]
        courses_ub: int = dataset["courses_per_period_ub"]
        course_load: List[int] = dataset["course_load"]
        prerequisites: List[List[int]] = dataset.get("prerequisites", [])
        # period[i]: index of the period to which course i is assigned, in [0, n_periods - 1]
        period_domains = [(0, self.n_periods - 1)] * self.n_courses
        # b[i, j]: boolean equal to 1 iff period[i] == j
        bool_domains = [(0, 1)] * (self.n_courses * self.n_periods)
        # load[j]: total load of period j
        load_domains = [(load_lb, load_ub)] * self.n_periods
        # max_load: maximum load over all periods (the objective)
        max_load_domain = [(load_lb, load_ub)]
        super().__init__(period_domains + bool_domains + load_domains + max_load_domain)
        self.bool_start = self.n_courses
        self.load_start = self.bool_start + self.n_courses * self.n_periods
        self.max_load = self.load_start + self.n_periods
        # link booleans to periods: b[i, j] <=> period[i] == j
        for i in range(self.n_courses):
            for j in range(self.n_periods):
                self.add_propagator(ALG_EQ_C_REIF, [self.bool(i, j), i], [j])
        # for each period j, compute its load: load[j] = sum_i course_load[i] * b[i, j]
        for j in range(self.n_periods):
            self.add_propagator(
                ALG_LINEAR_EQ_C,
                [self.bool(i, j) for i in range(self.n_courses)] + [self.load(j)],
                course_load + [-1, 0],
            )
        # max_load = max(load[j])
        self.add_propagator(ALG_MAX_EQ, [self.load(j) for j in range(self.n_periods)] + [self.max_load], [])
        # prerequisites: [a, b] means period[a - 1] < period[b - 1]
        for a, b in prerequisites:
            self.add_propagator(ALG_LEQ_C, [a - 1, b - 1], [-1])
        # for each period, bound the number of courses
        #    for j in range(self.n_periods):
        #       self.add_propagator(ALG_COUNT_GEQ_C, range(self.n_courses), [j, courses_lb])
        #       self.add_propagator(ALG_COUNT_LEQ_C, range(self.n_courses), [j, courses_ub])
        self.add_propagator(
            ALG_GCC, range(self.n_courses), [0] + [courses_lb] * self.n_periods + [courses_ub] * self.n_periods
        )

    def bool(self, i: int, j: int) -> int:
        """
        Returns the index of the boolean variable for (course i, period j).

        :param i: the course
        :type i: int
        :param j: the period
        :type j: int

        :return: the variable index
        :rtype: int
        """
        return self.bool_start + i * self.n_periods + j

    def load(self, j: int) -> int:
        """
        Returns the index of the load variable for period j.

        :param j: the period
        :type j: int

        :return: the variable index
        :rtype: int
        """
        return self.load_start + j

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns the solution as a dictionary mapping each period to its courses and total load.

        :param solution: the solution
        :type solution: NDArray

        :return: a printable representation of the solution
        :rtype: Any
        """
        periods: Dict[int, Dict[str, Any]] = {
            j: {"courses": [], "load": int(solution[self.load(j)])} for j in range(self.n_periods)
        }
        for i in range(self.n_courses):
            periods[int(solution[i])]["courses"].append(i + 1)
        return {"max_load": int(solution[self.max_load]), "periods": periods}
