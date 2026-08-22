# NuCS FlatZinc / MiniZinc adapter

This package lets you model in **MiniZinc** and solve with **NuCS**:

```bash
minizinc --solver nucs model.mzn
```

## How it works

MiniZinc flattens `model.mzn` into a `.fzn` file (a list of predicate calls) plus a `.ozn` output model.
The `fzn-nucs` executable (installed with NuCS) reads the `.fzn`, maps each builtin to a NuCS propagator,
solves with `BacktrackSolver`, and streams the FlatZinc solution format back to MiniZinc, which formats the
final output via `solns2out`/`.ozn`.

The globals library in `share/minizinc/nucs/` keeps the globals below native — either as a body-less
declaration or as a thin predicate delegating to a `nucs_`-prefixed one — so they reach NuCS's global
propagators instead of being decomposed into reified primitives:

```
all_different_int, at_least_int, at_most_int, bin_packing, bin_packing_capa, bin_packing_load, circuit,
count_eq, count_geq, count_leq, cumulative, diffn, disjunctive, disjunctive_strict, exactly_int,
global_cardinality_low_up, if_then_else_var_bool, increasing_int, inverse, lex_less_int, lex_lesseq_int,
nvalue, regular, strictly_increasing_int, subcircuit, table_bool, table_int, value_precede_chain_int,
value_precede_int
```

A global not listed there still works: MiniZinc decomposes it into builtins NuCS supports. Linear and
`element` constraints are standard FlatZinc builtins and are emitted natively by MiniZinc.

## Registering the solver

`fzn-nucs` is installed as a console script when you `pip install nucs`, so it is on your `PATH`.

To make MiniZinc discover the solver, point `MZN_SOLVER_PATH` at this `share` directory:

```bash
export MZN_SOLVER_PATH=$(python -c "import nucs.fzn, os; print(os.path.join(os.path.dirname(nucs.fzn.__file__), 'share'))")
minizinc --solver nucs model.mzn
```

Alternatively, copy `share/nucs.msc` into a MiniZinc user solver directory (and edit `mznlib` to an
absolute path to `share/minizinc/nucs`).

## Supported builtins

The `BUILTINS` registry in `builtins.py` dispatches these 97 FlatZinc builtins (the list is
checked against the registry by `tests/fzn/test_readme.py`, so it cannot drift):

```
all_different_int, array_bool_and, array_bool_element, array_bool_or, array_int_element, array_int_maximum,
array_int_minimum, array_var_bool_element, array_var_int_element, bool2int, bool_and, bool_clause, bool_eq,
bool_eq_reif, bool_ge_reif, bool_gt_reif, bool_le, bool_le_reif, bool_lin_eq, bool_lin_le, bool_lt,
bool_lt_reif, bool_not, bool_or, bool_xor, count_eq, count_geq, count_leq, decreasing_int,
fzn_all_different_int, fzn_count_eq, fzn_count_geq, fzn_count_leq, fzn_decreasing_int,
fzn_global_cardinality_low_up, fzn_increasing_int, fzn_lex_less_int, fzn_lex_lesseq_int, fzn_nvalue,
fzn_strictly_decreasing_int, fzn_strictly_increasing_int, fzn_value_precede_chain_int, fzn_value_precede_int,
global_cardinality_low_up, increasing_int, int_abs, int_div, int_eq, int_eq_imp, int_eq_reif, int_ge,
int_ge_reif, int_gt, int_gt_reif, int_le, int_le_imp, int_le_reif, int_lin_eq, int_lin_eq_imp,
int_lin_eq_reif, int_lin_ge, int_lin_ge_reif, int_lin_le, int_lin_le_imp, int_lin_le_reif, int_lin_ne,
int_lin_ne_reif, int_lt, int_lt_reif, int_max, int_min, int_mod, int_ne, int_ne_imp, int_ne_reif, int_plus,
int_times, lex_less_int, lex_lesseq_int, nucs_bin_packing_load, nucs_circuit, nucs_cumulative,
nucs_cumulative_var, nucs_diffn, nucs_disjunctive, nucs_if_then_else_var_bool, nucs_inverse, nucs_regular,
nucs_subcircuit, nucs_table_int, nvalue, set_in, set_in_reif, strictly_decreasing_int,
strictly_increasing_int, value_precede_chain_int, value_precede_int
```

Anything else raises a clear `FznUnsupportedError` naming the constraint. Coverage grows by adding one
entry to `nucs/fzn/builtins.py`.

### Known limitations

- NuCS domains are intervals: a non-contiguous set domain (`var {1, 3, 5}`) is bounded to its enclosing
  interval plus a `member` constraint for the holes. Filtering is bound-consistent throughout.
- No float constraints. `var set` is rewritten into an array of `var bool` by the standard `nosets.mzn`,
  which the library includes, so set models flatten rather than fail — but a set over a large universe
  blows up into a big boolean encoding.
- Boolean output variables are printed as `true`/`false`.
- Unbounded `var int` declarations fall back to a wide finite interval.
- `-t` (time limit) is accepted and ignored; MiniZinc enforces it by killing the process, so no
  `=====UNKNOWN=====` marker is produced.
