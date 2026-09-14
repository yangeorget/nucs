# FlatZinc benchmark models

Small MiniZinc models whose purpose is to *exercise the propagators the Python examples never post*.

Most of NuCS's globals are reachable only through FlatZinc — `value_precede_chain`, `bin_packing_load`,
`at_most`/`at_least` over a long array, `regular`, `nvalue`, `global_cardinality_low_up` — so
`nucs/examples` leaves them at zero calls. That gap repeatedly turned propagator work into a guess: a live
set for `count_leq_c` measured 1.5–4.4× on the propagator and had to be dropped because no model could say
whether that mattered, and the same happened to `diffn`. These models exist so that question has an answer.

Run them with:

```bash
NUMBA_CACHE_DIR=.numba/cache python scripts/fzn_benchmark.py --coverage
```

Each model declares the propagator it exists for as `% nucs-target: NAME` on its first line, and the report
prints that propagator's call count, no-change rate and arity. **A model that stops calling its target has
stopped earning its place** — MiniZinc rewrites, and a global can quietly lower to something else. That is
not hypothetical: `global_cardinality_low_up` with equal low and up bounds collapses entirely, and
`gcc_roster` posted zero of its intended target until its bounds were separated.

Compilation goes through `minizinc --solver nucs`, so it uses NuCS's own globals library and needs NuCS
registered (`fzn-nucs --register`). The `.fzn` is cached beside the `.mzn` and is not checked in.

| model | target | what it is |
|---|---|---|
| `bin_packing_load` | `BIN_PACKING_LOAD` | partition weighted items, minimising the heaviest bin |
| `count_shifts` | `COUNT_LEQ_C` | per-value quotas as `at_most`/`at_least` over a 72-long sequence |
| `diffn_packing` | `DIFFN` | strip packing, minimising the height |
| `gcc_roster` | `GCC` | a shift roster with per-shift cardinality bounds |
| `nvalue_assign` | `NVALUE` | minimise the number of distinct values in a constrained assignment |
| `regular_shifts` | `REGULAR` | a rostering pattern automaton over a long horizon |
| `value_precede_colouring` | `VALUE_PRECEDE` | graph colouring with the value symmetry broken |

Three of them — `count_shifts`, `value_precede_colouring` and `bin_packing_load` — post their target at a
useful **arity** (72, 50 and 42) but are not yet hot enough to time a change against; they are coverage,
not timing. `diffn_packing`, `gcc_roster`, `regular_shifts` and `nvalue_assign` are hot enough to A/B.
