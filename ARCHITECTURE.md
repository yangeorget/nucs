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

- **Interval domains only — bound consistency, no holes.** A domain is a single `(min, max)` `int32` pair; there are
  no sparse sets or bitmaps, and a propagator cannot remove a value from the middle of a domain. The trade: weaker
  pruning than arc consistency, but domains form a flat contiguous array that is O(1) to index and cheap to copy —
  which is what makes the choice-point and multiprocessing decisions below work.
- **Data-oriented state: all solver state lives in preallocated NumPy arrays.** No Python objects in the hot path:
  propagator metadata is laid out CSR-style (`bounds`, `propagator_variables`, `triggers` + `triggers_offsets`),
  statistics are an `int64` array, and everything is allocated once at solver init. Numba nopython mode forbids
  objects, and flat arrays make solver state trivially clonable. Consequence: jitted functions take many positional
  array arguments instead of a solver object — that is deliberate.
- **Choice points copy the whole domains array instead of trailing.** `cp_put` copies the top of `domains_stk`;
  backtracking is a stack-pointer decrement. O(variables) memcpy per choice point beats O(changes) trail bookkeeping
  because the copy is a contiguous `int32` memcpy and restore becomes free. The one exception is entailment, which
  *is* trailed (`entailed_propagator_depths` + `entailment_trail`): it is indexed by propagator, not variable, and
  entailment is monotonic within a branch, so a depth per propagator plus a depth-ordered trail to unwind is cheaper
  than copying a propagator-sized array at every choice point.
- **Propagators are stateless pure functions on a scratch buffer.** `compute_domains_*` receives a gathered copy of
  only its variables' domains (`domain_buffer`, sized once to the maximal arity) and mutates that copy;
  `update_domains` diffs it against the real domains to derive events. A propagator that detects inconsistency
  halfway through cannot corrupt global state, event computation is centralized in one place, and there is no
  per-propagator state to restore on backtrack.
- **Functions are values via numeric ids and wrapper addresses.** Propagators and heuristics register into typed
  lists indexed by `ALG_*` / heuristic ids; the ids live in integer arrays, and callables cross into nopython mode
  through `_get_wrapper_address` plus the `function_ptr_from_address` intrinsic (see `nucs/numba_helper.py`). Numba
  cannot dispatch on heterogeneous Python callables, so indirection through ids and addresses is the mechanism.
- **The propagation queue is a bucketed FIFO keyed by complexity.** `get_complexity_*` estimates a propagator's work
  per call; the queue (`nucs/buckets.py`) hashes that into 8 buckets by floor-log2, FIFO within a bucket, with
  membership flags for set semantics. Cheapest propagators run first, add and pop are O(1), no heap.
- **Parallelism is search-space splitting, not shared memory.** `MultiprocessingSolver` wraps N independent
  `BacktrackSolver`s over `problem.split(...)`, communicating only solutions through a queue. The GIL rules out
  shared-memory threading, and flat-array state makes cloning a subproblem cheap. There are no locks anywhere by
  construction.
- **The pure-Python escape hatch is a hard constraint.** Everything must also run under `NUMBA_DISABLE_JIT=1`
  (debugging, coverage, real tracebacks) — this is why `nucs/numba_helper.py` degrades typed lists to plain Python
  lists. Do not introduce Numba-only constructs without a non-JIT fallback.

## Explored, not adopted

- **A solver-owned scratch buffer for propagator working memory (benchmarked 2026-07, ~4%, shelved).** Some
  propagators need temporary arrays per call: `alldifferent` and `gcc` each do one `np.empty` inside
  `compute_domains_*` (already collapsed from several allocations into a single one). The explored alternative
  threads a second preallocated buffer — alongside `domain_buffer` — through the consistency algorithms and extends
  the propagator signature to `compute_domains_*(domains, parameters, scratch)`, removing the last allocator traffic
  from the propagation hot loop. Measured on queens 11–13 `solve_all`: a consistent 3.5–4.5% end-to-end speedup,
  matching a per-call microbenchmark saving of ~40 ns (the cost of one Numba NRT allocation), which is most visible
  at small arities where `alldifferent`'s O(n log n) body doesn't yet dominate. Two findings worth keeping:
  - **The scratch argument must be typed C-contiguous (`int32[::1]`) in `SIGN_COMPUTE_DOMAINS`.** Reusing the
    existing `parameters` array as scratch space avoids the signature change but is typed `int32[:]` (any layout),
    so every scratch slice loses compile-time contiguity and the hot loops pay strided indexing — that variant was
    5% *slower* than baseline despite allocating nothing. A local `np.empty` gives Numba layout knowledge for free;
    a passed-in buffer only matches it when the signature says `::1`.
  - **Why shelved:** `compute_domains_*` is a public extension point, so the third argument breaks every external
    propagator and every test that calls one directly — a 52-propagator, API-breaking change for a capped ~4-5% on
    alldifferent-heavy problems. With the single-allocation layout, Numba's allocator costs only ~40 ns per call;
    the current code is near-optimal without the break. Revisit if the signature changes anyway for another reason.
- **Incremental alldifferent via persistent warm permutations (benchmarked 2026-07, ~8% total, shelved with the
  scratch buffer).** Follow-up to the scratch-buffer experiment, using the same third argument: instead of pure
  scratch, each propagator gets a persistent per-propagator state row laid out as
  `[flag, min_sorted_vars[n], max_sorted_vars[n], scratch...]` (zeroed at solver init, `flag == 0` means cold since
  an all-zeros permutation is invalid). The insertion argsorts then warm-start from the previous call's
  permutations instead of rebuilding from the identity, costing O(n + inversions *since the previous call*) rather
  than displacement relative to index order. Findings:
  - **Hint state needs no backtrack bookkeeping.** A stale permutation is a valid input from any search node —
    merely a slower one — so nothing is trailed or restored, `problem.split(...)` clones just start cold, and the
    non-JIT fallback is untouched. Verified by identical solution counts across all configurations (73,712
    solutions of queens 13, millions of backtracks) and a bit-identical-propagation checksum in the
    microbenchmark. This is the cheapest sound form of propagator state; anything semantic (cached sums, counts)
    would instead need a generation stamp bumped on backtrack.
  - **Measured:** queens 11–13 `solve_all` ~7.5–8% end-to-end vs baseline (vs ~3.5% for scratch-only — warm
    permutations roughly double the win), langford(3,9) ~5.5%, all_interval(12) 0% (alldifferent is not its hot
    propagator). Per call, warm equals cold when sort keys correlate with variable index but removes the
    identity-seeded sort's O(n²) cliff when they don't: 2.4× at n=128, 14× at n=512, 49× at n=2048 (~1 ms/call cold vs
    21 µs warm). Mid-search, the no-offset alldifferent's keys (the values) decorrelate from index, which is where
    the end-to-end gain over scratch-only comes from; the diagonal constraints' monotone offsets keep their keys
    index-correlated, capping the gain on queens.
  - **Why shelved, and when to revisit:** same API break as the scratch buffer, so the same verdict — but if the
    `compute_domains_*` signature ever breaks for any reason, adopt this rather than plain scratch: same third
    argument, ~30 extra lines in `alldifferent` (and the same recipe applies to `gcc`), and it is insurance for
    FlatZinc-sourced models with large alldifferents over arbitrarily-ordered variables, which currently sit on
    the O(n²) sort cliff. Large-arity queens could not witness the effect either way: first-solution at n = 128
    with `VAR_HEURISTIC_SMALLEST_DOMAIN` is search-bound and did not finish under any configuration.