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
from numba import int32, int64, types  # type: ignore

START = 0  # index corresponding the start of a values range
END = 1  # index corresponding the end of a values range

MIN = 0  # min value of a domain
MAX = 1  # max value of a domain

PROP_INCONSISTENCY = 0  # returned by a propagator when inconsistent
PROP_CONSISTENCY = 1  # returned by a propagator when consistent
PROP_ENTAILMENT = 2  # returned by a propagator when entailed

PROBLEM_TO_FILTER = 0  # returned when the problem needs some filtering
PROBLEM_FILTERED = 1  # returned when the filtering of a problem has been completed
PROBLEM_INCONSISTENT = 2  # returned when the filtering of a problem detects an inconsistency
PROBLEM_SOLVED = 3  # returned when a problem is solved

SIGNATURE_COMPUTE_DOMAINS = int64(int32[:, :], int32[:])
SIGNATURE_DOM_HEURISTIC = int64(int32[:, :], int32[:, :])
SIGNATURE_VAR_HEURISTIC = int64(int32[:, :])

TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGNATURE_COMPUTE_DOMAINS)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGNATURE_DOM_HEURISTIC)
TYPE_VAR_HEURISTIC = types.FunctionType(SIGNATURE_VAR_HEURISTIC)
