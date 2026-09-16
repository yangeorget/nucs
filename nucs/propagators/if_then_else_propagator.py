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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_if_then_else(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables (2 * branches + 1)
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True)
def get_triggers_if_then_else(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever any bound changes.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index, unused here
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_if_then_else(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements the if-then-else selection y = x[k] where k is the smallest index such that the condition
    c[k] holds (the MiniZinc else branch is a literal-true condition, so a branch is normally always taken;
    when every condition is false nothing constrains y, matching the standard decomposition).

    The first b variables are the conditions c (booleans 0/1), the next b are the branch values x, and the
    last is the result y. The values may be any integers.

    The filtering reasons about candidates: a branch k is a candidate when it can still be the one taken, that
    is c[k] can hold, every earlier condition can fail, and x[k] and y can still be equal. Then:

    - c[i] cannot hold when no candidate precedes or is branch i, since the first true condition would name a
      branch that cannot be taken;
    - once some condition is fixed true a branch is taken, so there must be a candidate, y lies within the hull
      of the candidates' values, and a sole candidate is the taken branch: its condition holds, the earlier
      ones fail and its value equals y.

    Iterated to its fixpoint, this is bound-consistent on distinct variables. Narrowing y can remove candidates,
    which can leave a sole one, so one pass is not a fixpoint: the propagator is not idempotent, and the engine
    calls it again after a pass that changed a domain.

    :param domains: the domains of the variables, the b conditions then the b values then y
    :type domains: NDArray
    :param parameters: the parameters, unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = (len(domains) - 1) // 2
    y = domains[2 * b]
    candidate_nb = 0
    candidate = -1
    hull_min = y[DOMAIN_MAX]
    hull_max = y[DOMAIN_MIN]
    taken = False  # some condition is fixed true, so some branch is taken
    i = 0
    while i < b:
        c = domains[i]
        if c[DOMAIN_MAX] == 1:
            x = domains[b + i]
            low = max(x[DOMAIN_MIN], y[DOMAIN_MIN])
            high = min(x[DOMAIN_MAX], y[DOMAIN_MAX])
            if low <= high:
                candidate_nb += 1
                candidate = i
                hull_min = min(hull_min, low)
                hull_max = max(hull_max, high)
            elif candidate_nb == 0:
                # no branch up to i can be taken, so the first true condition cannot be c[i]
                if c[DOMAIN_MIN] == 1:
                    return PROP_INCONSISTENCY
                c[DOMAIN_MAX] = 0
            if c[DOMAIN_MIN] == 1:
                taken = True
                break  # the later branches can no longer be taken
        i += 1
    if not taken:
        # every condition may fail, leaving y unconstrained, so y and the values keep their bounds; the
        # conditions the scan has just closed cannot enable anything more
        return PROP_ENTAILMENT if candidate_nb == 0 else PROP_CONSISTENCY
    if candidate_nb == 0:
        return PROP_INCONSISTENCY
    y[DOMAIN_MIN] = max(y[DOMAIN_MIN], hull_min)
    y[DOMAIN_MAX] = min(y[DOMAIN_MAX], hull_max)
    if candidate_nb == 1:
        # the sole candidate is the taken branch
        for j in range(candidate):
            if domains[j, DOMAIN_MAX] == 1:
                domains[j, DOMAIN_MAX] = 0
        if domains[candidate, DOMAIN_MIN] == 0:
            domains[candidate, DOMAIN_MIN] = 1
        x = domains[b + candidate]
        x[DOMAIN_MIN] = max(x[DOMAIN_MIN], y[DOMAIN_MIN])
        x[DOMAIN_MAX] = min(x[DOMAIN_MAX], y[DOMAIN_MAX])
        if y[DOMAIN_MIN] == y[DOMAIN_MAX]:
            return PROP_ENTAILMENT  # the taken branch's value and y are equal and ground
    return PROP_CONSISTENCY  # not idempotent: after a pass that changed a domain, the engine calls it again
