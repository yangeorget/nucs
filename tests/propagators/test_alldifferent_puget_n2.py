from typing import List

from ncs.propagators.alldifferent_puget_n2 import AlldifferentPugetN2
from ncs.propagators.propagator import Propagator
from tests.propagators.alldifferent_abstract_test import AlldifferentAbstractTest


class TestAlldifferentPugetN2(AlldifferentAbstractTest):
    def all_different_propagator(self, variables: List[int]) -> Propagator:
        return AlldifferentPugetN2(variables)
