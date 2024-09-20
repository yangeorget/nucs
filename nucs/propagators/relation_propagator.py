from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_relation(n: int, data: NDArray) -> float:
    return 3 * len(data)


def get_triggers_relation(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_relation(domains: NDArray, data: NDArray) -> int:
    """
    :param domains: the domains of the variables
    :param data: the parameters of the propagator
    """
    n = len(domains)
    tuples = data.copy().reshape((-1, n))
    for domain_idx in range(n):
        tuples = tuples[
            (tuples[:, domain_idx] >= domains[domain_idx, MIN]) & (tuples[:, domain_idx] <= domains[domain_idx, MAX])
        ]
        if len(tuples) == 0:
            return PROP_INCONSISTENCY
    for domain_idx in range(n):  # no support for .min(axis=0) in Numba
        domains[domain_idx, MIN] = tuples[:, domain_idx].min()
        domains[domain_idx, MAX] = tuples[:, domain_idx].max()
    if len(tuples) == 1:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
