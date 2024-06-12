from typing import List

from ncs.propagators.alldifferent_lopez_ortiz import AlldifferentLopezOrtiz
from ncs.propagators.propagator import Propagator
from tests.alldifferent_abstract_test import AlldifferentAbstractTest


class TestAlldifferentLopezOrtiz(AlldifferentAbstractTest):
    def all_different_propagator(self, variables: List[int]) -> Propagator:
        return AlldifferentLopezOrtiz(variables)
