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

The globals library in `share/minizinc/nucs/` keeps `all_different`, `lex_lesseq` and
`global_cardinality_low_up` as native predicates (body-less declarations) so they reach NuCS's global
propagators instead of being decomposed. Linear and `element` constraints are standard FlatZinc builtins
and are emitted natively by MiniZinc.

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

`int_lin_eq`, `int_lin_le`, `int_eq`, `int_le`, `int_lt`, `int_plus`, `int_abs`,
`int_times` (with a constant operand), `int_max`, `int_min`, `all_different_int`,
`lex_lesseq_int`, `array_int_element`, `array_var_int_element`, `global_cardinality_low_up`
(contiguous cover). Unsupported builtins raise a clear error. Coverage grows by adding one entry to
`nucs/fzn/builtins.py`.

### Known limitations

- NuCS domains are intervals: a non-contiguous set domain (`var {1, 3, 5}`) is bounded to its enclosing
  interval plus a `member` constraint for the holes.
- No float constraints; set support is limited to `set_in` membership (no general set constraints).
- Boolean output variables are printed as `true`/`false`.
- Unbounded `var int` declarations fall back to a wide finite interval.
