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
from typing import Callable, List

import numpy as np
import pytest

from nucs.constants import PROP_INCONSISTENCY
from nucs.propagators.eq_c_imp_propagator import advise_eq_c_imp, compute_domains_eq_c_imp
from nucs.propagators.eq_c_reif_propagator import advise_eq_c_reif, compute_domains_eq_c_reif
from nucs.propagators.eq_imp_propagator import advise_eq_imp, compute_domains_eq_imp
from nucs.propagators.eq_reif_propagator import advise_eq_reif, compute_domains_eq_reif
from nucs.propagators.leq_c_imp_propagator import advise_leq_c_imp, compute_domains_leq_c_imp
from nucs.propagators.leq_c_reif_propagator import advise_leq_c_reif, compute_domains_leq_c_reif
from nucs.propagators.neq_c_reif_propagator import advise_neq_c_reif, compute_domains_neq_c_reif
from nucs.propagators.neq_imp_propagator import advise_neq_imp, compute_domains_neq_imp
from nucs.propagators.neq_reif_propagator import advise_neq_reif, compute_domains_neq_reif

# (advisor, compute_domains, number of operands after b, list of parameter arrays to try)
CS = [[-1], [0], [1], [2]]  # constant values for the *_c variants
NP: List[List[int]] = [[]]  # no parameter for the var/var variants
CASES = [
    (advise_eq_c_reif, compute_domains_eq_c_reif, 1, CS),
    (advise_neq_c_reif, compute_domains_neq_c_reif, 1, CS),
    (advise_eq_c_imp, compute_domains_eq_c_imp, 1, CS),
    (advise_eq_reif, compute_domains_eq_reif, 2, NP),
    (advise_neq_reif, compute_domains_neq_reif, 2, NP),
    (advise_eq_imp, compute_domains_eq_imp, 2, NP),
    (advise_neq_imp, compute_domains_neq_imp, 2, NP),
    (advise_leq_c_reif, compute_domains_leq_c_reif, 2, [[-1], [0], [1]]),
    (advise_leq_c_imp, compute_domains_leq_c_imp, 2, [[-1], [0], [1]]),
]


class TestAdvisors:
    @pytest.mark.parametrize("advise,compute,n_ops,param_sets", CASES)
    def test_advisor_is_sound(
        self, advise: Callable, compute: Callable, n_ops: int, param_sets: List[List[int]]
    ) -> None:
        # An advisor is SOUND iff it schedules (returns True) whenever compute_domains would change a domain
        # or report inconsistency. Skipping a run that would only entail (no domain change) is allowed.
        rng = range(-2, 4)
        op_boxes = [(lo, hi) for lo in rng for hi in rng if lo <= hi]
        for params in param_sets:
            p = np.array(params, dtype=np.int32)
            for b_dom in [[0, 0], [1, 1], [0, 1]]:
                for ops in _product_boxes(op_boxes, n_ops):
                    box = np.array([b_dom, *[list(o) for o in ops]], dtype=np.int32)
                    after = box.copy()
                    status = compute(after, p)
                    scheduled = advise(box.copy(), p)
                    must_schedule = not np.array_equal(box, after) or status == PROP_INCONSISTENCY
                    if must_schedule:
                        name = getattr(advise, "py_func", advise).__name__
                        assert scheduled, (
                            f"UNSOUND {name}: skipped a would-prune/inconsistency "
                            f"for domains={box.tolist()} params={params} (compute -> {after.tolist()}, status={status})"
                        )


def _product_boxes(op_boxes: List[tuple], n: int) -> List[tuple]:
    if n == 1:
        return [(b,) for b in op_boxes]
    return [(a, b) for a in op_boxes for b in op_boxes]
