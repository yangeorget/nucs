---
name: write-numba-friendly-python-code
description: Conventions for Numba-jitted code in NuCS — compile caching, what types jitted code may use, in-place array writes, passing functions as values, debugging compile errors, how to predict whether a change makes the solver faster, and how to measure it. Use whenever writing or editing an @njit function under nucs/, when a Numba typing or compilation error needs diagnosing, or when making the solver faster or judging a speed change.
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
- **`inline="always"` is not free on a hot path.** Numba inlines at its own IR level and increfs and decrefs every
  array the inlined function takes; those calls survive into the machine code. Look with `inspect_llvm` on a fresh
  compilation — a dispatcher loaded from the cache has no IR.
- **`bc_algorithm` is compiled with `_nrt=False`.** Code inlined into it must not allocate or index a typed list (it
  will not compile), and every jitted function it calls must be `inline="always"`, or LLVM inlines it with the
  refcounts of whichever context first put it in the cache. `tests/solvers/backtrack/test_bc_algorithm.py` checks both.
- **Debug a cryptic compile or typing error with `NUMBA_DISABLE_JIT=1`**: the plain-Python traceback points at the
  real line.

## Predict the speed before you build

Each rule below comes from a change that was built and measured. Read "Explored, not adopted" in ARCHITECTURE.md
before you propose a change for speed: it may be there already.

- **Remove calls before you make calls cheaper.** The cost of a model is mostly the number of propagator calls for
  each node. Change reporting (3x on `magic_sequence`) and an in-call fixpoint (`linear_eq_c`: 28-49% fewer calls)
  won; changes that only made each call cheaper gave approximately zero.
- **Do not allocate in a propagator.** Use the scratch buffer of the solver (ARCHITECTURE.md, "A solver-owned
  scratch buffer"). `regular` allocated two times in each call.
- **A new `@njit(cache=True)` function that a propagator calls costs about 0.2 s at the start of each process**, and
  nothing in the solve time. Make it `inline="always"`.
- **Do not reduce the number of parameters for speed.** From jitted code to jitted code, 29 array parameters cost the
  same as 5. At the boundary from Python to jitted code, each array costs about 23 ns, and that boundary is crossed
  once for each solution.
- **Do not expect speed from merging or shrinking small arrays.** The arrays that have one cell for each propagator
  are already in the L1 cache. Each merge measured zero or a loss: an `int32` record array was 4% slower, and the
  `uint8` `propagator_infos` (#213) was neutral. Make such a merge for clarity only, and predict zero. To expect
  speed, find a real cache miss first.
- **A sparse live set pays only when the candidate set becomes small and the arity is large.** Build it when less
  than 20% stays live; never when more than 40% stays live. It costs about 15 ns for each call, at all sizes.
- **A delta of the changes pays only when the propagator makes many calls for each filtering.** Predict it from the
  calls for each filtering, not from the size of the delta. Below approximately 3 calls, it loses (`count_eq`:
  0.75-0.80x).
- **Inline a callee of the propagation loop when LLVM does not.** `inline="always"` on `update_domains` gave 5.8%,
  because LLVM can then optimize across the two functions.

## Measure

Follow the protocol in `design/benchmarking.md`. The rules that are most frequently forgotten:

- Measure the solve time (`SOLVER_ELAPSED_TIME_MS`) and the wall time separately.
- Warm the Numba cache with one run that you discard. Clear it (`rm -rf .numba/cache`) when you change a constant
  that another module uses.
- Always run a control model that the change cannot affect, in the same command (`queens` for a change to one
  propagator).
- Nothing under 5% is a result. Run the two orders, and give each run an environment variable of random length.
- When the search trees are the same (`SOLVER_CHOICE_NB`), a difference in time is a cost for each node. A change
  to the pruning must decrease the number of nodes; if not, it does not do what it claims.
