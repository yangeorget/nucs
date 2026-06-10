# CLAUDE.md

Guidance for Claude Code working in this repository.

## Common commands

```bash
# Style (ruff)
./scripts/bash/style.sh

# All tests
NUMBA_CACHE_DIR=.numba/cache pytest

# Single file / single test
NUMBA_CACHE_DIR=.numba/cache pytest tests/examples/test_queens.py
NUMBA_CACHE_DIR=.numba/cache pytest tests/examples/test_queens.py::test_queens_4

# Debug or profile with pure Python (no JIT)
NUMBA_DISABLE_JIT=1 pytest tests/...
NUMBA_DISABLE_JIT=1 python -m "cProfile" -s time -m pytest tests/examples >> logs/examples.log

# Coverage
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest && coverage html

# FlatZinc adapter: register NuCS as a MiniZinc solver, then solve a .mzn model
fzn-nucs --register
minizinc --solver nucs model.mzn
```

`NUMBA_CACHE_DIR` is required for tests to share the JIT cache across runs. `NUMBA_DISABLE_JIT=1` falls back to
interpreted Python — slow, but tracebacks land in real source lines.

## Architecture

NuCS is a **Constraint Satisfaction Problem solver** that uses **Numba JIT** for performance. Pure Python, compiled at
runtime.

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

Constants worth knowing: `MIN`/`MAX` (domain row indices), `EVENT_MASK_*` (trigger flags), `PROP_*` (propagation result
codes), `STATS_IDX_*` (16 counters tracking backtracks, propagator calls, solutions, etc.).

## Numba rules

These apply to every `@njit` function in `nucs/` — most of the codebase.

- Use `@njit(cache=True, fastmath=True)`. `cache=True` is what makes warm starts fast; never remove it.
- No Python objects in jitted code: no `dict`, no exceptions, no `isinstance`, no strings other than literals. Pass
  typed `NDArray`s and ints.
- Mutate arrays in place (`domains[i][MIN] = ...`). Never rebind a slot (`domains[i] = ...`) — Numba can't always type
  that.
- Functions passed as values go through `_get_wrapper_address` (see `nucs/numba_helper.py`); the address is recovered to
  a typed callable at runtime. If you're adding a callable-typed parameter, this is the mechanism.
- When a Numba compile error is cryptic, re-run with `NUMBA_DISABLE_JIT=1` — the real Python traceback points to the
  line.

## Adding a propagator

Use the `/add-propagator` skill — it walks through the file layout, registration, and test pattern.

## Source file headers

Every file under `nucs/` and `tests/` starts with the ASCII-art copyright banner in `header.txt`. New files need it. To
re-stamp existing files:

```bash
addheader nucs -t header.txt
addheader tests -t header.txt
```

## Docstring format (Sphinx)

```python
"""
Returns the time complexity of the propagator as an int.

:param n: the number of variables
:type n: int
:param parameters: the parameters, unused here
:type parameters: NDArray

:return: an int
:rtype: int
"""
```

- Triple double-quotes on their own lines.
- One-line summary directly after the opening `"""`.
- `:param` / `:type` pair per parameter, in declaration order; `:return` / `:rtype` for the return.
- Descriptions are lowercase, no trailing period.

## Example

```python
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver


class QueensProblem(Problem):
    def __init__(self, n: int):
        super().__init__([(0, n - 1)] * n)
        self.add_propagator(ALG_ALLDIFFERENT, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(n))
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(0, -n, -1))


solver = BacktrackSolver(QueensProblem(8))
solution = next(solver.solve(), None)
```
