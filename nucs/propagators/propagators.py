import numpy as np
from numba import int32, int64, njit, types  # type: ignore
from numpy.typing import NDArray

from nucs.numba import NUMBA_DISABLE_JIT, build_function_address_list
from nucs.propagators.affine_eq_propagator import (
    compute_domains_affine_eq,
    get_complexity_affine_eq,
    get_triggers_affine_eq,
)
from nucs.propagators.affine_geq_propagator import (
    compute_domains_affine_geq,
    get_complexity_affine_geq,
    get_triggers_affine_geq,
)
from nucs.propagators.affine_leq_propagator import (
    compute_domains_affine_leq,
    get_complexity_affine_leq,
    get_triggers_affine_leq,
)
from nucs.propagators.alldifferent_propagator import (
    compute_domains_alldifferent,
    get_complexity_alldifferent,
    get_triggers_alldifferent,
)
from nucs.propagators.count_eq_propagator import (
    compute_domains_count_eq,
    get_complexity_count_eq,
    get_triggers_count_eq,
)
from nucs.propagators.dummy_propagator import compute_domains_dummy, get_complexity_dummy, get_triggers_dummy
from nucs.propagators.element_lic_propagator import (
    compute_domains_element_lic,
    get_complexity_element_lic,
    get_triggers_element_lic,
)
from nucs.propagators.element_liv_propagator import (
    compute_domains_element_liv,
    get_complexity_element_liv,
    get_triggers_element_liv,
)
from nucs.propagators.exactly_eq_propagator import (
    compute_domains_exactly_eq,
    get_complexity_exactly_eq,
    get_triggers_exactly_eq,
)
from nucs.propagators.lexicographic_leq_propagator import (
    compute_domains_lexicographic_leq,
    get_complexity_lexicographic_leq,
    get_triggers_lexicographic_leq,
)
from nucs.propagators.max_eq_propagator import compute_domains_max_eq, get_complexity_max_eq, get_triggers_max_eq
from nucs.propagators.max_leq_propagator import compute_domains_max_leq, get_complexity_max_leq, get_triggers_max_leq
from nucs.propagators.min_eq_propagator import compute_domains_min_eq, get_complexity_min_eq, get_triggers_min_eq
from nucs.propagators.min_geq_propagator import compute_domains_min_geq, get_complexity_min_geq, get_triggers_min_geq
from nucs.propagators.relation_propagator import (
    compute_domains_relation,
    get_complexity_relation,
    get_triggers_relation,
)

# The ordinals of the algorithms for all propagators (sorted by alphabetical ordering).
(
    ALG_AFFINE_EQ,
    ALG_AFFINE_GEQ,
    ALG_AFFINE_LEQ,
    ALG_ALLDIFFERENT,
    ALG_COUNT_EQ,
    ALG_DUMMY,
    ALG_ELEMENT_LIV,
    ALG_ELEMENT_LIC,
    ALG_EXACTLY_EQ,
    ALG_LEXICOGRAPHIC_LEQ,
    ALG_MAX_EQ,
    ALG_MAX_LEQ,
    ALG_MIN_EQ,
    ALG_MIN_GEQ,
    ALG_RELATION,
) = tuple(range(15))


GET_TRIGGERS_FCTS = [
    get_triggers_affine_eq,
    get_triggers_affine_geq,
    get_triggers_affine_leq,
    get_triggers_alldifferent,
    get_triggers_count_eq,
    get_triggers_dummy,
    get_triggers_element_liv,
    get_triggers_element_lic,
    get_triggers_exactly_eq,
    get_triggers_lexicographic_leq,
    get_triggers_max_eq,
    get_triggers_max_leq,
    get_triggers_min_eq,
    get_triggers_min_geq,
    get_triggers_relation,
]

GET_COMPLEXITY_FCTS = [
    get_complexity_affine_eq,
    get_complexity_affine_geq,
    get_complexity_affine_leq,
    get_complexity_alldifferent,
    get_complexity_count_eq,
    get_complexity_dummy,
    get_complexity_element_liv,
    get_complexity_element_lic,
    get_complexity_exactly_eq,
    get_complexity_lexicographic_leq,
    get_complexity_max_eq,
    get_complexity_max_leq,
    get_complexity_min_eq,
    get_complexity_min_geq,
    get_complexity_relation,
]


COMPUTE_DOMAINS_FCTS = [
    compute_domains_affine_eq,
    compute_domains_affine_geq,
    compute_domains_affine_leq,
    compute_domains_alldifferent,
    compute_domains_count_eq,
    compute_domains_dummy,
    compute_domains_element_liv,
    compute_domains_element_lic,
    compute_domains_exactly_eq,
    compute_domains_lexicographic_leq,
    compute_domains_max_eq,
    compute_domains_max_leq,
    compute_domains_min_eq,
    compute_domains_min_geq,
    compute_domains_relation,
]

COMPUTE_DOMAIN_SIGNATURE = int64(int32[:, :], int32[:])
COMPUTE_DOMAIN_TYPE = types.FunctionType(COMPUTE_DOMAIN_SIGNATURE)
COMPUTE_DOMAINS_ADDRS = (
    np.array(build_function_address_list(COMPUTE_DOMAINS_FCTS, COMPUTE_DOMAIN_SIGNATURE))
    if not NUMBA_DISABLE_JIT
    else np.empty(0)
)


@njit(cache=True)
def pop_propagator(triggered_propagators: NDArray, not_entailed_propagators: NDArray, previous_prop_idx: int) -> int:
    for prop_idx, triggered_prop in enumerate(triggered_propagators):
        if triggered_prop and not_entailed_propagators[prop_idx] and prop_idx != previous_prop_idx:
            triggered_propagators[prop_idx] = False
            return prop_idx
    return -1
