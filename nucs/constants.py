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
import os

from numba import bool, int32, int64, types, uint8, uint16  # type: ignore

START = 0  # index corresponding the start of a values range
END = 1  # index corresponding the end of a values range

MIN = 0  # min value of a domain
MAX = 1  # max value of a domain

PROP_INCONSISTENCY = 0  # returned by a propagator when inconsistent
PROP_CONSISTENCY = 1  # returned by a propagator when consistent
PROP_ENTAILMENT = 2  # returned by a propagator when entailed

PROBLEM_INCONSISTENT = 0  # returned when the filtering of a problem detects an inconsistency
PROBLEM_UNBOUND = 1  # returned when the filtering of a problem has been completed but the problem is not solved
PROBLEM_BOUND = 2  # returned when a problem is solved

SIGNATURE_COMPUTE_DOMAINS = int64(int32[:, :], int32[:])  # domains, parameters
SIGNATURE_DOM_HEURISTIC = int64(int32[:], int32[:])  # shr_domain, shr_domain_copy
SIGNATURE_VAR_HEURISTIC = int64(int32[:, :])  # shr_domains
SIGNATURE_CONSISTENCY_ALG = int64(
    int64[:],  # statistics
    uint8[:],  # algorithms
    uint16[:, :],  # var_bounds
    uint16[:, :],  # param_bounds
    uint16[:],  # dom_indices_arr
    int32[:],  # dom_offsets_arr
    uint16[:],  # props_dom_indices
    int32[:, :],  # props_dom_offsets
    int32[:],  # props_parameters
    bool[:, :, :],  # shr_domains_propagators
    int32[:, :, :],  # shr_domains_stack
    bool[:, :],  # not_entailed_propagators_stack
    uint16[:, :],  # dom_update_stack
    uint8[:],  # stacks_height
    bool[:],  # triggered_propagators
    int64[:],  # compute_domains_addrs
)

TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGNATURE_COMPUTE_DOMAINS)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGNATURE_DOM_HEURISTIC)
TYPE_VAR_HEURISTIC = types.FunctionType(SIGNATURE_VAR_HEURISTIC)
TYPE_CONSISTENCY_ALG = types.FunctionType(SIGNATURE_CONSISTENCY_ALG)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")
