.. _heuristics:

**********
Heuristics
**********

NUCS provides heuristics for selecting a variable and for selecting a value (more generally, reducing its domain):

Heuristics for selecting a variable
###################################

NUCS provides the following functions for selecting a variable

.. autofunction:: nucs.heuristics.variable.critical_resource_var_heuristic.critical_resource_var_heuristic
.. autofunction:: nucs.heuristics.variable.dom_wdeg_var_heuristic.dom_wdeg_var_heuristic
.. autofunction:: nucs.heuristics.variable.first_not_instantiated_var_heuristic.first_not_instantiated_var_heuristic
.. autofunction:: nucs.heuristics.variable.greatest_domain_var_heuristic.greatest_domain_var_heuristic
.. autofunction:: nucs.heuristics.variable.largest_maximal_value_var_heuristic.largest_maximal_value_var_heuristic
.. autofunction:: nucs.heuristics.variable.max_regret_var_heuristic.max_regret_var_heuristic
.. autofunction:: nucs.heuristics.variable.min_earliest_start_var_heuristic.min_earliest_start_var_heuristic
.. autofunction:: nucs.heuristics.variable.smallest_domain_var_heuristic.smallest_domain_var_heuristic
.. autofunction:: nucs.heuristics.variable.smallest_minimal_value_var_heuristic.smallest_minimal_value_var_heuristic


Heuristics for reducing the chosen domain
#########################################

NUCS provides the following functions for reducing a domain.

.. autofunction:: nucs.heuristics.domain.max_value_dom_heuristic.max_value_dom_heuristic
.. autofunction:: nucs.heuristics.domain.mid_value_dom_heuristic.mid_value_dom_heuristic
.. autofunction:: nucs.heuristics.domain.min_cost_dom_heuristic.min_cost_dom_heuristic
.. autofunction:: nucs.heuristics.domain.min_value_dom_heuristic.min_value_dom_heuristic
.. autofunction:: nucs.heuristics.domain.random_value_dom_heuristic.random_value_dom_heuristic
.. autofunction:: nucs.heuristics.domain.split_low_dom_heuristic.split_low_dom_heuristic
.. autofunction:: nucs.heuristics.domain.split_high_dom_heuristic.split_high_dom_heuristic

