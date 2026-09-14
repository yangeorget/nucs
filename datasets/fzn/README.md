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
| `bin_packing_load` | `BIN_PACKING_LOAD` | partition 54 weighted items into 7 bins, minimising the heaviest |
| `count_shifts` | `COUNT_LEQ_C` | per-shift quotas as `at_most`/`at_least` over a 120-long roster |
| `diffn_packing` | `DIFFN` | strip packing, minimising the height |
| `gcc_roster` | `GCC` | a shift roster with per-shift cardinality bounds |
| `nvalue_assign` | `NVALUE` | minimise the number of distinct values in a constrained assignment |
| `regular_shifts` | `REGULAR` | a rostering pattern automaton over a long horizon |
| `value_precede_colouring` | `VALUE_PRECEDE` | graph colouring with the value symmetry broken |

Six are hot enough to A/B a change against — `count_shifts` (2.9M `count_leq_c` calls at arity 120, with
2.1M `count_geq_c` beside it), `gcc_roster` (3.7M `gcc`), `regular_shifts` (544k `regular`),
`diffn_packing` (64k `diffn`), `nvalue_assign` (19k `nvalue`) and `bin_packing_load` (13k calls at arity
61, where the propagator is most of the runtime because its cost is quadratic in that arity).

`value_precede_colouring` is the exception, and it is **coverage only by nature rather than by neglect**.
`value_precede(s, t)` returns entailment as soon as the earliest position that can hold `s` is ground to
`s`, and entailment is trailed, so any search that grounds variables settles it at shallow depth and it
stays settled for the whole subtree. Five shapes were tried — fixed prefix, fixed suffix, reversed search
order, all-solutions enumeration to 945,937 nodes, and barring the chained values from an early window —
and the propagator is called at most 12 times in any of them. Keeping it live needs `s` excluded from a
long prefix without ever being ground there, and arranging that makes the model infeasible, because
`value_precede` then bars `t` from the same prefix and the cascade runs out of values. **There is nothing
in that propagator to optimise**, which is worth knowing and is what this model records.

Two knobs make a model hot without making it huge. `% nucs-solve: all` enumerates every solution instead
of stopping at the first, which is what turns a model the solver satisfies greedily into one whose search
depth follows the instance size. And pre-assigning part of the problem sets the propagators' **arity**
independently of how much search is left, so a model can be wide — which is what makes it representative —
without its enumeration exploding. `count_shifts` uses both.
