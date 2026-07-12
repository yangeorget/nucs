# Check style

Style is enforced with ruff (lint + format) and mypy. Run the whole thing with:

```bash
./scripts/bash/style.sh
```

This runs, over `nucs` and `tests`:

```bash
ruff check --fix nucs tests   # lint, auto-fixing what it can
ruff format nucs tests        # format
mypy nucs tests               # type-check
```

Run it after editing any Python file under `nucs/` or `tests/`, and make sure it passes
before committing. No `NUMBA_CACHE_DIR` is needed — style checks don't run the JIT.
