# CLAUDE.md

Guidance for Claude Code working in this repository.
For repository layout and core concepts: see ARCHITECTURE.md.

## Common commands

```bash
# Style (ruff & mypy)
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

`NUMBA_CACHE_DIR` is required for tests to share the JIT cache across runs.
`NUMBA_DISABLE_JIT=1` falls back to interpreted Python — slow, but tracebacks land in real source lines.

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
