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
"""
The arrays of the search: the backtrackable state and its views, the trail and the stack of choice points, laid out
and sized for a problem.
"""

import numpy as np
from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.solvers.choice_points import CHOICE_POINT_WIDTH

# Trail entries a step of the search needs beyond one per cell of the backtrackable state.
# The barrier in trail_set trails each cell at most once per choice point, so a fixpoint cannot need more
# than len(state) entries however long it runs. The tightenings the search applies around it are not
# covered by that budget: each writes at a mark the trail holds nothing for yet, so every one of their
# writes is trailed. A tightening writes a domain's two bounds and, when it grounds the variable, the
# unbound count; a step applies at most two of them -- branch's decision is one, while backtracking a
# choice point applies its parked alternative and then the branch-and-bound objective bound.
TIGHTENING_TRAIL_ENTRY_NB = 3  # the two bounds of a domain and the unbound count
STEP_TIGHTENING_NB = 2  # an alternative then the objective bound, the longer of the two ways out of a step


def allocate_search_arrays(
    problem: Problem,
) -> tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Lays out the backtrackable state of a problem and allocates the trail and the stack of choice points.

    :param problem: the problem, already initialized
    :type problem: Problem

    :return: the arrays of the search, in this order: all the backtrackable state; the domains, a (domain_nb, 2) view
             of its head; the entailment flags, a view of its middle; the trail entries that any one step of the
             search can need; the undo log of (cell index, old value) pairs; the trail size, as a one-cell array;
             the index of the last trail entry of each cell, -1 for none; the stack of choice points; the height of
             the stack, as a one-cell array
    :rtype: Tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    # all the backtrackable state in one flat int32 array, so that one undo log and one undo loop
    # restore every kind of it:
    #     [ 2 * domain_nb domain bounds | propagator_nb entailment flags | propagator state | the unbound count ]
    # domains is a (domain_nb, 2) view of its head and entailed a view of the middle -- the same
    # memory, addressed the way each reader wants it -- so the flat index of (variable, bound) is
    # (variable << 1) | bound, and that of propagator p is 2 * domain_nb + p. Restoring a domain
    # bound and reactivating an entailed propagator are then the same instruction. Propagator state
    # (problem.state_width cells, addressed per propagator by offsets[:, OFFSETS_STATE], see
    # Problem.init_propagator_arrays) sits between the entailment flags and the unbound count so that
    # unbound_index() -- len(state) - 1 -- stays independent of it; every state block in this stage
    # defaults its root value to 0 (np.zeros below), which is what every propagator using one expects.
    domain_nb = problem.domain_nb
    propagator_nb = problem.propagator_nb
    propagator_entailment_offset = 2 * domain_nb
    propagator_state_offset = propagator_entailment_offset + propagator_nb
    unbound_count_offset = propagator_state_offset + problem.state_width
    state = np.zeros(unbound_count_offset + 1, dtype=np.int32)
    domains = state[:propagator_entailment_offset].reshape(domain_nb, 2)
    entailed = state[propagator_entailment_offset:propagator_state_offset]
    # the guard lets a choice point trail each *trailable* cell at most once -- every domain bound, every
    # entailment flag, the count, and the trailed prefix of each propagator's state block -- so a fixpoint
    # cannot need more than that many entries, whatever it does; the tightenings the search applies around
    # it write at their own mark and are counted on top. The solver grows the trail when this much room is
    # no longer there. Counted rather than taken as len(state), which would include the untrailed hint
    # suffixes: bc_algorithm only ever trails block p's prefix below offsets[p, OFFSETS_STATE_HINT], so a
    # hint cell can never reach the trail, and a wide alldifferent would otherwise inflate trail_log -- 16x
    # trail_headroom rows of 2 int32 -- by 128 bytes per scratch cell it can never use.
    trailable_nb = propagator_state_offset + problem.state_trailed_width + 1
    trail_headroom = trailable_nb + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB
    # Both starting sizes are a flat floor for the models the flat floor already covers, and a
    # model-derived one for the wide models it does not reach. The flat halves are measured: across the
    # 27 benchmark models the trail never exceeds 4096 entries and the stack never exceeds 64 rows, so
    # 65536 and 8192 leave every one of them growing exactly zero times -- raising them further would buy
    # nothing and give back the memory trailing was introduced to win.
    # The model-derived halves are where growth actually happens. A live trail runs between 2 and 12
    # times headroom on those same models, so 16 covers the observed band with margin; a stack deep
    # enough to matter is bounded by the decisions on the path, which scales with domain_nb. Both bind
    # only past a few thousand variables -- exactly where a doubling copies the most, and where the
    # allocation is small next to the triggers and propagator arrays a model that wide already carries.
    trail_log = np.empty((max(1 << 16, 16 * trail_headroom), 2), dtype=np.int32)
    trail_top = np.zeros((1,), dtype=np.int32)
    trail_indices = np.full(len(state), -1, dtype=np.int32)
    choice_point_stk = np.zeros((max(1 << 13, 4 * domain_nb), CHOICE_POINT_WIDTH), dtype=np.int32)
    choice_point_top = np.ones((1,), dtype=np.uint32)
    return (
        state,
        domains,
        entailed,
        trail_headroom,
        trail_log,
        trail_top,
        trail_indices,
        choice_point_stk,
        choice_point_top,
    )
