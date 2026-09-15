---
name: write-numba-friendly-python-code
description: Conventions for Numba-jitted code in NuCS — compile caching, what types jitted code may use, in-place array writes, passing functions as values, and debugging compile errors. Use whenever writing or editing an @njit function under nucs/, or when a Numba typing or compilation error needs diagnosing.
---

# Write Numba-friendly Python code

Most of `nucs/` runs under `@njit`. In that code:

- **Decorate with `@njit(cache=True)`.** The on-disk cache is what makes warm starts fast; never drop `cache=True`.
- **Use only typed NDArrays and scalars.** No `dict`, no exceptions, no `isinstance`, no strings other than literals.
- **Write array cells in place** (`domains[i, DOMAIN_MIN] = ...`). Never rebind a slot (`domains[i] = ...`): Numba
  cannot always type it.
- **Pass functions as addresses.** `addresses_from_functions` in `nucs/numba_helper.py` turns functions into addresses
  with `_get_wrapper_address`, and `function_ptr_from_address` recovers a typed callable at run time. Use this
  mechanism for any new callable-typed parameter.
- **Debug a cryptic compile or typing error with `NUMBA_DISABLE_JIT=1`**: the plain-Python traceback points at the
  real line.
