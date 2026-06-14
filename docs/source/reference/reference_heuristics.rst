.. _heuristics:

**********
Heuristics
**********

NUCS provides heuristics for selecting a variable and for selecting a value (more generally, reducing its domain):

Heuristics for selecting a variable
###################################

NUCS provides the following functions for selecting a variable

.. autofunction:: nucs.heuristics.first_not_instantiated_var_heuristic.first_not_instantiated_var_heuristic
.. autofunction:: nucs.heuristics.greatest_domain_var_heuristic.greatest_domain_var_heuristic
.. autofunction:: nucs.heuristics.largest_maximal_value_var_heuristic.largest_maximal_value_var_heuristic
.. autofunction:: nucs.heuristics.max_regret_var_heuristic.max_regret_var_heuristic
.. autofunction:: nucs.heuristics.smallest_domain_var_heuristic.smallest_domain_var_heuristic
.. autofunction:: nucs.heuristics.smallest_minimal_value_var_heuristic.smallest_minimal_value_var_heuristic


Heuristics for reducing the chosen domain
#########################################

NUCS provides the following functions for reducing a domain.

.. autofunction:: nucs.heuristics.max_value_dom_heuristic.max_value_dom_heuristic
.. autofunction:: nucs.heuristics.mid_value_dom_heuristic.mid_value_dom_heuristic
.. autofunction:: nucs.heuristics.min_cost_dom_heuristic.min_cost_dom_heuristic
.. autofunction:: nucs.heuristics.min_value_dom_heuristic.min_value_dom_heuristic
.. autofunction:: nucs.heuristics.split_low_dom_heuristic.split_low_dom_heuristic
.. autofunction:: nucs.heuristics.split_high_dom_heuristic.split_high_dom_heuristic

