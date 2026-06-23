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
import json

from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.examples.square.square_problem import SquarePlacementProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.square --dataset datasets/examples/square/square_21.json
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--dataset", default="datasets/examples/square/square_21.json")
    args = parser.parse_args()
    with open(args.dataset, "r") as json_file:
        dataset = json.load(json_file)
    problem = SquarePlacementProblem(dataset["width"], dataset["height"], dataset["squares"])
    kwargs = solver_kwargs_from_args(args, searches=problem.recommended_searches())
    solver = BacktrackSolver(problem, **kwargs)
    solution = next(solver.solve(), None)
    if args.display_stats:
        solver.print_statistics()
    if args.display_solutions and solution is not None:
        for i in range(problem.square_nb):
            print(f"square {problem.sizes[i]}: ({int(solution[problem.x(i)])}, {int(solution[problem.y(i)])})")
