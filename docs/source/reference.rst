## Reference documentation
### Limits
Domains limits are 32-bits integers.
The number of variables is an unsigned 16-bits integer.

### Propagators
NUCS currently provides the following propagators:
- `affine_eq_propagator`
- `affine_geq_propagator`
- `affine_leq_propagator`
- `alldifferent_propagator`
- `count_eq_propagator`
- `element_lic_propagator`
- `element_liv_propagator`
- `exactly_eq_propagator`
- `lexicographic_leq_propagator`
- `max_eq_propagator`
- `max_leq_propagator`
- `min_eq_propagator`
- `min_geq_propagator`
- `relation_propagator`

### Heuristics
#### Functions for selecting a shared domain
- `first_not_instantiated_var_heuristic`: selects the first non-instantiated shared domain
- `last_not_instantiated_var_heuristic`: selects the last non-instantiated shared domain
- `smallest_domain_var_heuristic`: selects the smallest shared domain which is not instantiated
- `greatest_domain_var_heuristic`: selects the greatest shared domain which is not instantiated
-
#### Functions for reducing the chosen shared domain
- `min_value_dom_heuristic`: selects the minimal value of the domain
- `max_value_dom_heuristic`: selects the maximal value of the domain
- `split_low_dom_heuristic`: selects the first half of the domain





