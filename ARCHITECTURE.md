# NuCS architecture

NuCS is a **Constraint Satisfaction Problem solver** that uses **Numba JIT** for performance.
Pure Python, compiled at runtime.

## Repository structure

- **`nucs/problems/`** — a `Problem` carries `domains` (one `(min, max)` per variable; bound when `min == max`) and a
  list of propagators added via `add_propagator(ALG_*, *variable_index_iterables, parameters=...)`.
- **`nucs/propagators/`** — one file per constraint, plus `propagators.py` which registers each as a numeric `ALG_*` id.
  Each propagator is three functions: `compute_domains_*` (filtering, returns `PROP_INCONSISTENCY` /
  `PROP_CONSISTENCY` / `PROP_ENTAILMENT`), `get_triggers_*` (when to re-wake), `get_complexity_*` (queue ordering). See
  `nucs/propagators/abs_eq_propagator.py` for the minimal template.
- **`nucs/solvers/`** — `BacktrackSolver` (backtracking + propagation) and `MultiprocessingSolver` (wraps several
  `BacktrackSolver`s over `problem.split(...)`). Iterate solutions with `solver.solve()`, or call
  `solver.minimize(var)` / `solver.maximize(var)`.
- **`nucs/heuristics/`** — variable heuristics pick the next unbound variable, domain heuristics pick the next value to
  try. Both are Numba-jitted with signatures in `nucs/constants.py`.
- **`nucs/fzn/`** — the **FlatZinc adapter**: model in MiniZinc, solve with NuCS via `minizinc --solver nucs`. Pipeline
  is `parser.py` (FlatZinc text → IR) → `model.py` (`FznModel` builds a `Problem`) → `builtins.py` (the `BUILTINS`
  dispatch table: FlatZinc builtin name → `add_propagator` calls) → `runner.py` (solve) → `output.py` (FlatZinc solution
  stream). `fzn-nucs` is the console script MiniZinc invokes; `fzn-nucs --register` writes the solver config into
  `~/.minizinc/solvers`. `share/minizinc/nucs/` is the globals library that keeps selected globals (alldifferent, gcc,
  lex, table) native instead of decomposed. Grow coverage by adding one entry to `BUILTINS` (and, for a kept global, one
  predicate file under `share/minizinc/nucs/`) — see the `/add-propagator` skill, step 7.

## Constants

Constants worth knowing: `MIN`/`MAX` (domain row indices), `EVENT_MASK_*` (trigger flags), `PROP_*` (propagation result
codes), `STATS_IDX_*` (16 counters tracking backtracks, propagator calls, solutions, etc.).

## Important decisions