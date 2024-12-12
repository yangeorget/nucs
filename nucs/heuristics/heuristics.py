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
from typing import Callable

from nucs.heuristics.first_not_instantiated_var_heuristic import first_not_instantiated_var_heuristic
from nucs.heuristics.greatest_domain_var_heuristic import greatest_domain_var_heuristic
from nucs.heuristics.max_regret_var_heuristic import max_regret_var_heuristic
from nucs.heuristics.max_value_dom_heuristic import max_value_dom_heuristic
from nucs.heuristics.mid_value_dom_heuristic import mid_value_dom_heuristic
from nucs.heuristics.min_cost_dom_heuristic import min_cost_dom_heuristic
from nucs.heuristics.min_value_dom_heuristic import min_value_dom_heuristic
from nucs.heuristics.smallest_domain_var_heuristic import smallest_domain_var_heuristic
from nucs.heuristics.split_high_dom_heuristic import split_high_dom_heuristic
from nucs.heuristics.split_low_dom_heuristic import split_low_dom_heuristic

VAR_HEURISTIC_FCTS = []
DOM_HEURISTIC_FCTS = []


def register_var_heuristic(var_heuristic_fct: Callable) -> int:
    """
    Register a variable heuristic by adding it function to the corresponding list of functions.
    :param var_heuristic_fct: a function that implements the variable heuristic
    :return: the index of the variable heuristic
    """
    VAR_HEURISTIC_FCTS.append(var_heuristic_fct)
    return len(VAR_HEURISTIC_FCTS) - 1


def register_dom_heuristic(dom_heuristic_fct: Callable) -> int:
    """
    Register a domain heuristic by adding it function to the corresponding list of functions.
    :param dom_heuristic_fct: a function that implements the domain heuristic
    :return: the index of the domain heuristic
    """
    DOM_HEURISTIC_FCTS.append(dom_heuristic_fct)
    return len(DOM_HEURISTIC_FCTS) - 1


VAR_HEURISTIC_FIRST_NOT_INSTANTIATED = register_var_heuristic(first_not_instantiated_var_heuristic)
VAR_HEURISTIC_GREATEST_DOMAIN = register_var_heuristic(greatest_domain_var_heuristic)
VAR_HEURISTIC_MAX_REGRET = register_var_heuristic(max_regret_var_heuristic)
VAR_HEURISTIC_SMALLEST_DOMAIN = register_var_heuristic(smallest_domain_var_heuristic)

DOM_HEURISTIC_MAX_VALUE = register_dom_heuristic(max_value_dom_heuristic)
DOM_HEURISTIC_MID_VALUE = register_dom_heuristic(mid_value_dom_heuristic)
DOM_HEURISTIC_MIN_COST = register_dom_heuristic(min_cost_dom_heuristic)
DOM_HEURISTIC_MIN_VALUE = register_dom_heuristic(min_value_dom_heuristic)
DOM_HEURISTIC_SPLIT_HIGH = register_dom_heuristic(split_high_dom_heuristic)
DOM_HEURISTIC_SPLIT_LOW = register_dom_heuristic(split_low_dom_heuristic)
