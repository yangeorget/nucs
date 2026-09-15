---
name: run-tests
description: Runs the NuCS pytest suite with the Numba JIT on (the default) or off — the whole suite, one file or one test, and debugging, profiling and coverage runs. Use when running or debugging tests, when a failure inside jitted code needs a real traceback, or when profiling or measuring coverage.
---

# Run tests

Tests run under pytest from the repository root. The Numba JIT is on by default.

```bash
# All tests (JIT on)
pytest

# One file / one test
pytest tests/examples/test_queens.py
pytest tests/examples/test_queens.py::TestQueens::test_solve_all

# No JIT (debugging)
NUMBA_DISABLE_JIT=1 pytest tests/...

# Profile in pure Python (no JIT)
NUMBA_DISABLE_JIT=1 python -m cProfile -s time -m pytest tests/examples >> logs/examples.log

# Coverage (no JIT)
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest && coverage html
```

- `.claude/settings.json` sets `NUMBA_CACHE_DIR=.numba/cache` for every session, so runs share one compiled cache.
  Outside Claude Code, set it yourself.
- `NUMBA_DISABLE_JIT=1` runs interpreted Python: slow, but tracebacks land on real source lines. Use it when a failure
  inside jitted code is unreadable.
- `tests/fzn/test_minizinc.py` runs the *installed* `fzn-nucs` through MiniZinc, not the working tree. After changing
  `nucs/`, reinstall before trusting it: `rm -rf build && pip install --no-deps .`.
