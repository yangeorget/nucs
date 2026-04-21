# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

```bash
# Check code style (uses ruff)
./scripts/bash/style.sh

# Run all tests (must set PYTHONPATH and NUMBA_CACHE_DIR)
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest

# Run a single test file
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/examples/test_queens.py

# Run a single test
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/examples/test_queens.py::test_queens_4

# Compute code coverage
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest
coverage html

# Measure the performance
NUMBA_DISABLE_JIT=1 python -m "cProfile" -s time -m nucs.examples.queens | more

# Build the pip package
python -m build

# Publish the package
python -m twine upload --verbose dist/*

# Fix source file headers
addheader nucs -t header.txt
addheader tests -t header.txt

# Generate documentation
sphinx-build -M html docs/source docs/output
```

## High-Level Architecture

NuCS is a **Constraint Satisfaction Problem (CSP) solver** that uses **Numba JIT compilation** for performance.
It is 100% Python but compiles hot paths to machine code at runtime.

### Core Components

**1. Problems** (`nucs/problems/`)

- A `Problem` is defined by **domains** (possible values for variables) and **propagators** (constraints).
- Domain values are stored as `(min, max)` tuples. When min equals max, the variable is bound.
- Subclass `Problem` to define new problems, then call `add_propagator()` to add constraints.

**2. Propagators** (`nucs/propagators/`)

- Each propagator implements constraint propagation logic using three Numba-jitted functions:
    - `compute_domains()`: Prunes inconsistent values from domains
    - `get_triggers()`: Returns when the propagator should be triggered (on min change, max change, or ground)
    - `get_complexity()`: Estimates computational cost for sorting propagators
- Propagators are registered in `nucs/propagators/propagators.py` using `register_propagator()` which assigns them a
  unique algorithm ID (e.g., `ALG_ALLDIFFERENT`).
- Propagation returns: `PROP_INCONSISTENCY` (failure), `PROP_CONSISTENCY` (pruned), or `PROP_ENTAILMENT` (done).

**3. Solvers** (`nucs/solvers/`)

- `BacktrackSolver`: Main solver using backtracking search with constraint propagation
- `MultiprocessingSolver`: Parallel solver that splits the problem across multiple processes
- Uses choice points for state restoration and a propagation queue to manage triggered propagators

**4. Heuristics** (`nucs/heuristics/`)

- **Variable heuristics** select which unbound variable to branch on next (e.g., smallest domain first)
- **Domain heuristics** select which value to try first for a chosen variable (e.g., min value, mid value)
- Heuristics are Numba-jitted functions with signatures defined in `nucs/constants.py`.

### Key Technical Details

**Numba Integration**

- Heavy use of `@njit(cache=True, fastmath=True)` for performance-critical code
- Function pointers are passed via `_get_wrapper_address` from `numba.experimental.function_type`
- See `nucs/numba_helper.py` for the mechanism to recover functions from addresses at runtime
- Set `NUMBA_CACHE_DIR` to cache compiled code between runs
- Set `NUMBA_DISABLE_JIT=1` to disable JIT (useful for profiling/debugging)

**Consistency Algorithms**

- `CONSISTENCY_ALG_BC`: Bound consistency (default) - propagates on bound changes
- Shaving variants try to reduce domains by testing values

**Statistics**

- 16 statistics counters defined in `nucs/constants.py` (indices `STATS_IDX_*`)
- Track propagator calls, backtracks, solutions, search space size, etc.

### Example Problem Structure

```python
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver


class QueensProblem(Problem):
    def __init__(self, n):
        super().__init__([(0, n - 1)] * n)  # n variables with domains [0, n-1]
        self.add_propagator(ALG_ALLDIFFERENT, range(n))  # columns
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(n))  # diagonals
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(0, -n, -1))  # anti-diagonals


problem = QueensProblem(8)
solver = BacktrackSolver(problem)
solution = solver.find_one()
```

### File Organization Patterns

- `nucs/examples/*/`: Each example is a self-contained module with a `*_problem.py` and `__main__.py`
- `nucs/propagators/*_propagator.py`: Individual propagator implementations
- Tests mirror the source structure: `tests/examples/`, `tests/propagators/`, etc.
- Documentation source is in `docs/source/` using Sphinx/rst format
