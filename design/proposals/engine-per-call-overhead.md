# Plan: the engine's per-call overhead

> **Status:** closed, 2026-09-15. Written and measured the same day: the dispatch was worth 2.5% and the statistics
> 1.4%, so the stop rule fired. The results are in `ARCHITECTURE.md`, after *Devirtualising the dispatch*.

## The finding that prompts it

`golomb(11)` runs **37.3M propagator calls in 4,530 ms — 121 ns a call**. Its busiest propagator, `sum_eq`,
accounts for 26.7M of those (72%), and pricing its two passes by neutral duplication puts its **whole body
at 8.7–11.5%** of the model. So roughly **110 ns of every 121 ns is the machinery around the propagator**,
not the propagator.

That is the target. It is worth attacking because it is not specific to golomb: any model built from many
small constraints pays it per call, and FlatZinc lowers almost everything into `int_lin_*`, `*_reif` and
other arity-2/3 propagators — `gcc_roster` makes 13.3M `linear_neq_c` calls at arity 2, `regular_shifts`
2.5M `eq_c_imp` calls at arity 2.

## What a call does today

Per iteration of `bc_algorithm`'s loop, for a propagator of arity `a` with `t` trailed cells:

| step | cost shape |
|---|---|
| `buckets_pop` | scan from the cached lowest bucket, ~6 array accesses |
| statistics | 2 int64 read-modify-writes (2 more on the no-change exit) |
| offsets reads | 5 loads from `offsets[prop_idx]` and `offsets[prop_idx + 1]` |
| gather | `a` × (1 load of `propagator_variables`, 2 loads of `state`, 2 stores to `domain_buffer`) |
| trailing prefix | `t` × `trail_set`; zero iterations for most propagators, but two offsets loads regardless |
| **dispatch** | indirect call through `compute_domains_fcts[algorithm]`, plus two array slices built as arguments |
| the propagator | ~10 ns at arity 3 |
| `update_domains` | `a` × (1 load, 2 loads, a compare, `tighten_at`), then per changed variable a walk of `triggers[lo:hi]` — **mean slice 4.8 on golomb, 3.0 on queens** — each testing membership and entailment, with `buckets_add` on a hit |

## Hypotheses, in the order I would test them

### H1 — the indirect dispatch dominates *(highest expected value)*

`compute_domains_fcts[algorithm](...)` is a call through a Numba typed function list. It cannot be inlined,
it forces live values to memory across the boundary, and it blocks every cross-boundary optimisation. For a
body that is ~10 ns of arithmetic, that is plausibly the largest single item.

**What makes this promising and not merely plausible:** real models dispatch over almost nothing.
`golomb(11)` posts 101 propagators over **3 distinct algorithms**; `queens(12)` posts 3 over **1**. A
dispatch that tested the two or three algorithms a problem actually uses, falling through to the function
list for the rest, would let LLVM inline the hot bodies outright.

- **Test**: a throwaway build with `if algorithm == ALG_SUM_EQ: compute_domains_sum_eq(...)` ahead of the
  indirect call, for golomb's three. Measure golomb(10) and (11); `queens` is the control (it would take the
  `alldifferent` branch, so use `magic_square` or `langford` instead — a model whose algorithms are *not* in
  the chain).
- **Predicts**: if H1 is right, golomb improves by more than the 8.7–11.5% the whole `sum_eq` body is worth,
  because the win is on `leq_c` and `alldifferent` too.
- **Falsified if**: under 5%. Then the dispatch is cheap and the cost is spread across the other steps.
- **If it works**, the design question is how to generate the chain without hard-coding: a per-problem
  specialised `bc_algorithm` is the honest version, and Numba can compile one per call-site signature.
  That is a real design cost and should not be paid before the measurement justifies it.

### H2 — `update_domains`' write-back and trigger walk

Already partly known: inlining `update_domains` into the loop was worth 5.8%, and change reporting — which
skips it entirely — was worth 3.0× on magic_sequence. But magic_sequence has arity 201; at arity 3 the
write-back is three variables and the trigger walk ~5 entries.

- **Test**: it cannot be duplicated (it mutates and schedules). Price it instead by comparing a model's
  reporting propagators against the same model with reporting switched off — that difference *is*
  `update_domains` for the calls that skip it. `schur_lemma` (97.1% no-change on `sum_leq_c`) is the
  sharpest instrument here.
- **Predicts**: a share that grows with arity, so small on golomb and large on wide models. If so, this is
  already harvested by change reporting and there is nothing new.

### H3 — the statistics counters

Four int64 read-modify-writes per call on the common paths, 37M times on golomb. They are also the reason
every A/B in this repo is checkable, so they cannot simply go.

- **Test**: a build with every `statistics[...] += 1` removed is semantically neutral for the search and
  gives the ceiling directly. One run.
- **Predicts**: 2–5%. If it is more, the counters are worth gating; Numba specialises on a literal boolean,
  so a `statistics_on` flag closed over at compile time would cost nothing at run time.
- **Note**: the per-algorithm tail (`STATS_MAX + STATS_ALG_WIDTH * algorithm`) touches a *different* cache
  line per algorithm. With 3 algorithms that is 3 extra lines resident; harmless here, possibly not in a
  FlatZinc model using 20.

### H4 — the gather into `domain_buffer`

`a` scattered reads out of `state` and `2a` stores into the buffer, then the propagator reads the buffer and
`update_domains` reads it again. The copy exists so `compute_domains_*` sees a dense `(a, 2)` array.

- Already optimised once: assigning rows was replaced by scalar stores, 90 ms → 55 ms on magic_sequence.
- The remaining idea — letting a propagator index `state` directly through its variable list — changes
  `compute_domains_*`'s signature, which is a **public extension point**. Do not start here. Revisit only if
  H1 and H3 come back small and the gather measures large.

## Do not retry

Measured negative already, with the numbers in `ARCHITECTURE.md`:

- recording which variables changed and handing the propagator the delta (0.75–0.80×, the append costs
  31–40% while the shorter scan gives back 5–6%);
- a cache keyed on the exact domains (0 hits in 131 lookups — the event that wakes a propagator is the event
  that invalidates it);
- merging or shrinking the hot arrays (`triggers`, `offsets`, the domains) — every variant measured zero,
  because they are L1-resident already.

## Measurement protocol

Moved to [design/benchmarking.md](../benchmarking.md), which keeps it up to date for all the measurements.

## Stop rules

- If Part 0 (H3's neutral build plus H1's hard-coded chain) shows nothing above 10% between them, stop and
  write it up: the 110 ns is diffuse, and diffuse overhead is not worth a specialised engine.
- If H1 is large, the follow-up is a design question — per-problem specialisation of `bc_algorithm` — and
  should be planned separately rather than grown from the spike.
