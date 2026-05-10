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

from nucs.examples.car_sequencing.car_sequencing_problem import CarSequencingProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.car_sequencing
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--dataset", default="datasets/car_sequencing/ecai88.json")
    args = parser.parse_args()
    with open(args.dataset, "r") as json_file:
        dataset = json.load(json_file)
        problem = CarSequencingProblem(dataset)
        run_solver(BacktrackSolver(problem, **solver_kwargs_from_args(args)), args)
