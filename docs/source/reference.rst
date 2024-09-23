#######################
Reference documentation
#######################

***********
Propagators
***********

NUCS currently provides the following propagators:

- :code:`affine_eq_propagator`
- :code:`affine_geq_propagator`
- :code:`affine_leq_propagator`
- :code:`alldifferent_propagator`
- :code:`count_eq_propagator`
- :code:`element_lic_propagator`
- :code:`element_liv_propagator`
- :code:`exactly_eq_propagator`
- :code:`lexicographic_leq_propagator`
- :code:`max_eq_propagator`
- :code:`max_leq_propagator`
- :code:`min_eq_propagator`
- :code:`min_geq_propagator`
- :code:`relation_propagator`

**********
Heuristics
**********

Functions for selecting a shared domain
#######################################

- :code:`first_not_instantiated_var_heuristic`: selects the first non-instantiated shared domain
- :code:`last_not_instantiated_var_heuristic`: selects the last non-instantiated shared domain
- :code:`smallest_domain_var_heuristic`: selects the smallest shared domain which is not instantiated
- :code:`greatest_domain_var_heuristic`: selects the greatest shared domain which is not instantiated

Functions for reducing the chosen shared domain
###############################################
- :code:`min_value_dom_heuristic`: selects the minimal value of the domain
- :code:`max_value_dom_heuristic`: selects the maximal value of the domain
- :code:`split_low_dom_heuristic`: selects the first half of the domain





