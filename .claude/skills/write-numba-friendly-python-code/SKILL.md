---
name: write-numba-friendly-python-code
description: Skill to write Numba friendly Python code. Use it whenever you write or edit an `@njit` function in `nucs/`.
---

# Write Numba friendly Python code

These apply to every `@njit` function in `nucs/` — most of the codebase.

- Use `@njit(cache=True)`. `cache=True` is what makes warm starts fast; never remove it.
- No Python objects in jitted code: no `dict`, no exceptions, no `isinstance`, no strings other than literals. Pass
  typed `NDArray`s and ints.
- Mutate arrays in place (`domains[i][MIN] = ...`). Never rebind a slot (`domains[i] = ...`) — Numba can't always type
  that.
- Functions passed as values go through `_get_wrapper_address` (see `nucs/numba_helper.py`); the address is recovered to
  a typed callable at runtime. If you're adding a callable-typed parameter, this is the mechanism.
- When a Numba compile error is cryptic, re-run with `NUMBA_DISABLE_JIT=1` — the real Python traceback points to the
  line.
