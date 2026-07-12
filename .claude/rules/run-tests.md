# Run tests

Tests run under pytest. The Numba JIT is on by default; disable it for debugging or profiling.

```bash
# All tests (JIT on)
NUMBA_CACHE_DIR=.numba/cache pytest

# Single file / single test
NUMBA_CACHE_DIR=.numba/cache pytest tests/examples/test_queens.py
NUMBA_CACHE_DIR=.numba/cache pytest tests/examples/test_queens.py::test_queens_4

# No JIT (debugging)
NUMBA_DISABLE_JIT=1 pytest tests/...

# Profile with pure Python (no JIT)
NUMBA_DISABLE_JIT=1 python -m "cProfile" -s time -m pytest tests/examples >> logs/examples.log

# Coverage (no JIT)
NUMBA_DISABLE_JIT=1 PYTHONPATH=. coverage run --source=nucs,tests -m pytest && coverage html
```

- `NUMBA_CACHE_DIR=.numba/cache` is required for JIT runs to share the compiled cache across runs.
- `NUMBA_DISABLE_JIT=1` falls back to interpreted Python — slow, but tracebacks land in real source lines.
