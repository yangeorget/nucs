from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN

OPERATOR_EQ = 0
OPERATOR_GEQ = 1
OPERATOR_LEQ = 2


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    n = len(domains)
    # TODO: case where data[1:] is 0
    domain_sum = np.full(2, data[0], dtype=np.int32)
    for i in range(n):
        ai = data[i + 1]
        domain_sum[MIN] -= domains[i, MIN if ai < 0 else MAX] * ai
        domain_sum[MAX] -= domains[i, MAX if ai < 0 else MIN] * ai
    new_domains = domains.copy()
    operator = data[-1]
    if operator == OPERATOR_EQ:
        for i in range(n):
            ai = data[i + 1]
            if ai < 0:
                new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai))
            elif ai > 0:
                new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (domain_sum[MIN] // -ai))
            if new_domains[i, MIN] > domains[i, MAX]:
                return None
            if ai < 0:
                new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (-domain_sum[MIN] // -ai))
            elif ai > 0:
                new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // ai))
            if new_domains[i, MAX] < domains[i, MIN]:
                return None
    elif operator == OPERATOR_LEQ:
        for i in range(n):
            ai = data[i + 1]
            if ai < 0:
                new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai))
                if new_domains[i, MIN] > domains[i, MAX]:
                    return None
            elif ai > 0:
                new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // ai))
                if new_domains[i, MAX] < domains[i, MIN]:
                    return None
    elif operator == OPERATOR_GEQ:
        for i in range(n):
            ai = data[i + 1]
            if ai > 0:
                new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (domain_sum[MIN] // -ai))
                if new_domains[i, MIN] > domains[i, MAX]:
                    return None
            if ai < 0:
                new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (-domain_sum[MIN] // -ai))
                if new_domains[i, MAX] < domains[i, MIN]:
                    return None
    return new_domains
