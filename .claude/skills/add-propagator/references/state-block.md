# State block: `get_state_<name>` and `reports_changes`

`compute_domains` receives `prop_state`, its slice of one solver-owned int32 array. The slice is empty unless the
propagator declares a size:

```python
def get_state_<name>(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    # plain Python, called once per propagator at problem init
    # returns (trailed_nb, hint_nb): a trailed prefix then an untrailed hint suffix, as one contiguous block
```

Name the cells with module constants that say which part each lives in, as `count_leq_c_propagator.py` does:

```python
STATE_LIVE_NB = 0  # trailed: how many x_i are still undetermined
STATE_COUNT_MIN = 1  # trailed: how many x_i are already fixed to a
STATE_REPORT = 2  # the engine's change-report cell, which must be the first cell of the hint suffix
STATE_LIVE = 3  # the live permutation
```

## Kinds of cell

- **Scratch** (hint suffix): replaces a per-call `np.empty`/`np.zeros`, which would be paid at every fixpoint. Fully
  overwrite it before reading it. `gcc` uses one.
- **Hint** (hint suffix): carried across calls but never trailed, so it holds whatever the last call from *any* node
  left. Store only values that stay valid however stale — staleness may cost time, never correctness.
  `alldifferent` warm-starts its sort permutations from one, which is sound because a stale permutation is still a
  permutation.
- **Per-node state** (trailed prefix): saved and restored like a domain bound, so a node reads back what it wrote.
  Each trailed cell costs trail entries, so keep the prefix narrow. The `count_*` propagators trail a live-set size
  and a count there; `lexleq` and `relation` also trail cells.

## Change reporting: `reports_changes=True`

When calls often narrow nothing, the engine can skip its write-back scan — but only if the propagator says so. Opt in
with `reports_changes=True` and reserve the report cell, the first cell of the hint suffix: `prop_state[trailed_nb]`,
so `hint_nb >= 1`.

- The engine sets the cell to 1 ("changed") before every call and reads it only when the call returns
  `PROP_CONSISTENCY`. Write 0 only when the call narrowed no domain, as in `sum_eq_c_propagator.py`.
- **A wrong 0 silently drops a pruning**: the search explores more, or returns a wrong answer if the dropped write
  was the one that failed the node. Forgetting to write the cell only costs the scan, so when unsure leave it at 1.
- `assert_compute_domains` checks the report on every call of your test cases.

## Rules

- **The block is zeroed once, at solver init** — not on backtrack, not on an `OPTIM_RESET` restart. If a call needs
  a cleared block, clear it in `compute_domains`.
- **Warm-starting does not bound the work.** `argsort_into_warm` costs the inversions since the previous call: small
  down a descent, O(n²) after a jump. That is why `argsort_into` keeps its `np.argsort` fallback above `SORT_MAX_N`.
  If a hint pays off only with locality, keep an unconditional fallback.
- **Justify every cell in the `get_state_<name>` docstring**: why it is trailed, or why being untrailed is safe — see
  `get_state_alldifferent`, `get_state_gcc` and `get_state_sum_eq_c`. A wrong untrailed claim shows up as a
  heisenbug: the search finds different solutions depending on the path it took to a node.

## Tests

- **Trailed live set**: `assert_live_set_is_sound(compute_domains_<name>, domains, parameters, rng, backtrack)` runs a
  chain of narrowings — with `backtrack=True`, also a restore of the trailed prefix — and checks at each step that the
  warm block agrees with a cold one on status, bounds and change report. See `tests/propagators/test_count_eq.py`.
- **Change reporting** needs no extra test: `assert_compute_domains` checks it.
