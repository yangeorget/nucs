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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


class PropagatorTest:
    def assert_compute_domains(
        self,
        compute_domains_fct: Callable,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        domains_arr = np.array(
            [(domain, domain) if isinstance(domain, int) else domain for domain in domains], dtype=np.int32
        )
        assert compute_domains_fct(domains_arr, np.array(parameters, dtype=np.int32)) == consistency_result
        if expected_domains:
            assert np.all(domains_arr == np.array(expected_domains))
