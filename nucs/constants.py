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
