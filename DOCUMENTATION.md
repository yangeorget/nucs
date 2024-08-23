# NUCS

## Why Python ?
NUCS is a Python library leveraging Numpy and Numba.

Python is a powerful and flexible programing language that allows to express complex problems in a few lines of code.

Numpy brings the computational power of languages like C and Fortran to Python, a language much easier to learn and use.

Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. 
Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

## Architecture
### Variables
Variables are simply integers used to reference domains.
Here is the case of three variables with domains [1, 10].

| Variable | Domain    |
|----------|-----------|
| 0        | $[1, 10]$ | 
| 1        | [1, 10]   |
| 2        | [1, 10]   |

### Domains
NUCS only support integer domains.

### Propagators (aka Constraints)

## Reference documentation
### Propagators
NUCS currently provides the following propagators:
- `affine_eq_propagator`
- `affine_geq_propagator`
- `affine_leq_propagator`
- `alldifferent_propagator`
- `count_eq_propagator`
- `exactly_eq_propagator`
- `lexicographic_leq_propagator`
- `max_eq_propagator`
- `max_leq_propagator`
- `min_eq_propagator`
- `min_geq_propagator`

### Heuristics
NUCS currently provides the following heuristics:
- `first_not_instantiated_var_heuristic`: selects the first non instantiated variable
- `smallest_domain_var_heuristic`: selects the variable with the smallest domain which is not instantiated
- `min_value_dom_heuristic`: selects the minimal value of the domain 
- `split_low_dom_heuristic`: selects the first half of the domain






