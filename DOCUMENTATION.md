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

> Variables are thus implicit.

Let's consider three variables with integer domains **[1, 10]**.

| Variable index | Domain      |
|----------------|-------------|
| **0**          | **[1, 10]** | 
| **1**          | **[1, 10]** |
| **2**          | **[1, 10]** |

NUCS could represent these domains as a `numpy.ndarray` of shape `(3,2)`: 
- one row for each domain
- the first column corresponds to the minimal values
- the second column corresponds to the maximal values

The reality is a bit more complex, as explained below.

### Domains
#### Shared domains
Let's consider two variables with initial domains **[1, 10]** and the constraint **v_0 = v_1 + 4**.
Because of bound consistency, the domain of **v_0** is **[5, 10]** and the domain of **v_1** is **[1, 6]**
There is a lot of overhead here: 
- each time we update the domain of **v_0**, we also need to update the domain of **v_1** (and vice versa)
- from the point of view of the variable choice heuristic, 
the problem is made arbitrarily large and it makes no difference whether **v_0** or **v_1** is selected

We can efficiently replace both domains by a single shared domain and two offsets **4** and **0**:

| Variable index | Offset | Shared domain index |
|----------------|--------|---------------------|
| **0**          | **4**  | **0**               | 
| **1**          | **0**  | **0**               |

With:

| Shared domain index | Shared domain |
|---------------------|---------------|
| **0**               | **[1, 6]**    | 

Note that if there was nothing to share, we would simply have :

| Variable index | Offset | Shared domain index |
|----------------|--------|---------------------|
| **0**          | **0**  | **0**               | 
| **1**          | **0**  | **1**               |

> Domains are also implicit.

Because of shared domains and offsets, the constructor of `Problem` accepts 3 arguments:
- `shr_domains`: the shared domains as a list of pairs of integers (if the minimal and maximal values of the pair are equal, the pair can be replaced by the value)
- `dom_indices`: a list of integers representing, for each variable, the index of its shared domain 
- `dom_offsets`: a list of integers representing, for each variable, the offset of its shared domain

Internally, shared domains, domain indices and offsets are stored using `numpy.ndarray`.
#### Integer domains
NUCS only support integer domains.

Boolean domains are simply integer domains of the form **[0, 1]**.

### Propagators (aka Constraints)
Propagators are defined by two functions:
- `compute_domains(domains: NDArray, data: NDArray) -> int`: this function updates the domains (the actual domains, not the shared ones) of the variables of the propagator. 
It is expected to implement bound consistencyt and to be idempotent (a second consecutive run should not update the domains).
It returns a status: `PROP_INCONSISTENCY`, `PROP_CONSISTENCY` or `PROP_ENTAILMENT`.
- `get_triggers(size: int, data: NDArray) -> NDArray`: this function returns a `numpy.ndarray` of shape `(size, 2)`. 
Let `triggers` be such an array, `triggers[i, MIN] == True` means that the propagator should be triggered whenever the minimum value of variable `ì` changes.

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






